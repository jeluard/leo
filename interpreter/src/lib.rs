// Copyright (C) 2019-2024 Aleo Systems Inc.
// This file is part of the Leo library.

// The Leo library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The Leo library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the Leo library. If not, see <https://www.gnu.org/licenses/>.

use leo_ast::{Ast, Node as _, NodeBuilder};
use leo_errors::{CompilerError, InterpreterHalt, LeoError, Result, emitter::Handler};
use leo_span::{Span, source_map::FileName, symbol::with_session_globals};

use snarkvm::prelude::TestnetV0;

use colored::*;
use std::{
    collections::HashMap,
    fmt::Display,
    fs,
    path::{Path, PathBuf},
};

mod util;
use util::*;

mod cursor;
use cursor::*;

mod value;
use value::*;

/// Contains the state of interpretation, in the form of the `Cursor`,
/// as well as information needed to interact with the user, like
/// the breakpoints.
struct Interpreter {
    cursor: Cursor<'static>,
    actions: Vec<InterpreterAction>,
    handler: Handler,
    node_builder: NodeBuilder,
    breakpoints: Vec<Breakpoint>,
    filename_to_program: HashMap<PathBuf, String>,
}

#[derive(Clone, Debug)]
struct Breakpoint {
    program: String,
    line: usize,
}

#[derive(Clone, Debug)]
enum InterpreterAction {
    LeoInterpretInto(String),
    LeoInterpretOver(String),
    RunFuture(usize),
    Breakpoint(Breakpoint),
    Into,
    Over,
    Step,
    Run,
}

impl Interpreter {
    fn new<'a, P: 'a + AsRef<Path>>(source_files: impl IntoIterator<Item = &'a P>) -> Result<Self> {
        Self::new_impl(&mut source_files.into_iter().map(|p| p.as_ref()))
    }

    fn get_ast(path: &Path, handler: &Handler, node_builder: &NodeBuilder) -> Result<Ast> {
        let text = fs::read_to_string(path).map_err(|e| CompilerError::file_read_error(path, e))?;
        let filename = FileName::Real(path.to_path_buf());
        let source_file = with_session_globals(|s| s.source_map.new_source(&text, filename));
        leo_parser::parse_ast::<TestnetV0>(handler, node_builder, &text, source_file.start_pos)
    }

    fn new_impl(source_files: &mut dyn Iterator<Item = &Path>) -> Result<Self> {
        let handler = Handler::default();
        let node_builder = Default::default();
        let mut cursor: Cursor<'_> = Cursor::new(
            true, // really_async
        );
        let mut filename_to_program = HashMap::new();
        for path in source_files {
            let ast = Self::get_ast(path, &handler, &node_builder)?;
            // TODO: This leak is silly.
            let ast = Box::leak(Box::new(ast));
            for (&program, scope) in ast.ast.program_scopes.iter() {
                filename_to_program.insert(path.to_path_buf(), program.to_string());
                for (name, function) in scope.functions.iter() {
                    cursor.functions.insert(GlobalId { program, name: *name }, function);
                }

                for (name, composite) in scope.structs.iter() {
                    cursor.structs.insert(
                        GlobalId { program, name: *name },
                        composite.members.iter().map(|member| member.identifier.name).collect(),
                    );
                }

                for (name, _mapping) in scope.mappings.iter() {
                    cursor.mappings.insert(GlobalId { program, name: *name }, HashMap::new());
                }

                for (name, const_declaration) in scope.consts.iter() {
                    cursor.frames.push(Frame {
                        step: 0,
                        element: Element::Expression(&const_declaration.value),
                        user_initiated: false,
                    });
                    cursor.over()?;
                    let value = cursor.values.pop().unwrap();
                    cursor.globals.insert(GlobalId { program, name: *name }, value);
                }
            }
        }

        Ok(Interpreter {
            cursor,
            handler,
            node_builder,
            actions: Vec::new(),
            breakpoints: Vec::new(),
            filename_to_program,
        })
    }

    fn action(&mut self, act: InterpreterAction) -> Result<Option<Value>> {
        use InterpreterAction::*;

        let ret = match &act {
            RunFuture(n) => {
                let future = self.cursor.futures.remove(*n);
                for async_exec in future.0.into_iter().rev() {
                    self.cursor.values.extend(async_exec.arguments);
                    self.cursor.frames.push(Frame {
                        step: 0,
                        element: Element::DelayedCall(async_exec.function),
                        user_initiated: true,
                    });
                }
                self.cursor.step()?
            }
            LeoInterpretInto(s) | LeoInterpretOver(s) => {
                let s = s.trim();
                if s.ends_with(';') {
                    let statement = leo_parser::parse_statement::<TestnetV0>(&self.handler, &self.node_builder, s)
                        .map_err(|_e| {
                            LeoError::InterpreterHalt(InterpreterHalt::new("failed to parse statement".into()))
                        })?;
                    // TODO: This leak is silly.
                    let stmt = Box::leak(Box::new(statement));
                    self.cursor.frames.push(Frame { step: 0, element: Element::Statement(stmt), user_initiated: true });
                } else {
                    let expression = leo_parser::parse_expression::<TestnetV0>(&self.handler, &self.node_builder, s)
                        .map_err(|_e| {
                            LeoError::InterpreterHalt(InterpreterHalt::new("failed to parse expression".into()))
                        })?;
                    // TODO: This leak is silly.
                    let expr = Box::leak(Box::new(expression));
                    expr.set_span(Default::default());
                    self.cursor.frames.push(Frame {
                        step: 0,
                        element: Element::Expression(expr),
                        user_initiated: true,
                    });
                };

                if matches!(act, LeoInterpretOver(..)) { self.cursor.over()? } else { self.cursor.step()? }
            }

            Step => self.cursor.whole_step()?,

            Into => self.cursor.step()?,

            Over => self.cursor.over()?,

            Breakpoint(breakpoint) => {
                self.breakpoints.push(breakpoint.clone());
                StepResult { finished: false, value: None }
            }

            Run => {
                while !self.cursor.frames.is_empty() {
                    if let Some((program, line)) = self.current_program_and_line() {
                        if self.breakpoints.iter().any(|bp| bp.program == program && bp.line == line) {
                            return Ok(None);
                        }
                    }
                    self.cursor.step()?;
                }
                StepResult { finished: false, value: None }
            }
        };

        self.actions.push(act);

        Ok(ret.value)
    }

    fn view_current(&self) -> Option<impl Display> {
        if let Some(span) = self.current_span() {
            if span != Default::default() {
                return with_session_globals(|s| s.source_map.contents_of_span(span));
            }
        }

        Some(match self.cursor.frames.last()?.element {
            Element::Statement(statement) => format!("{statement}"),
            Element::Expression(expression) => format!("{expression}"),
            Element::Block { block, .. } => format!("{block}"),
            Element::DelayedCall(gid) => format!("Delayed call to {gid}"),
        })
    }

    fn view_current_in_context(&self) -> Option<impl Display> {
        let span = self.current_span()?;
        if span == Default::default() {
            return None;
        }
        with_session_globals(|s| {
            let source_file = s.source_map.find_source_file(span.lo)?;
            let first_span = Span::new(source_file.start_pos, span.lo);
            let last_span = Span::new(span.hi, source_file.end_pos);
            Some(format!(
                "{}{}{}",
                s.source_map.contents_of_span(first_span)?,
                s.source_map.contents_of_span(span)?.red(),
                s.source_map.contents_of_span(last_span)?,
            ))
        })
    }

    fn current_program_and_line(&self) -> Option<(String, usize)> {
        if let Some(span) = self.current_span() {
            if let Some(location) = with_session_globals(|s| s.source_map.span_to_location(span)) {
                let line = location.line_start;
                if let FileName::Real(name) = &location.source_file.name {
                    if let Some(program) = self.filename_to_program.get(name) {
                        return Some((program.clone(), line));
                    }
                }
            }
        }
        None
    }

    fn current_span(&self) -> Option<Span> {
        self.cursor.frames.last().map(|f| f.element.span())
    }
}

const INSTRUCTIONS: &str = "
This is the Leo Interpreter. You probably want to start by running a function or transition.
For instance
#into program.aleo/main()
Once a function is running, commands include
#into    to evaluate into the next expression or statement;
#step    to take one step towards evaluating the current expression or statement;
#over    to complete evaluating the current expression or statement;
#run     to finish evaluating
#quit    to exit the interpreter.

You can set a breakpoint with
#break program_name line_number

You may also use one letter abbreviations for these commands, such as #i.

You may simply enter expressions or statements on the command line
to evaluate. For instance, if you want to see the value of a variable w:
w
If you want to set w to a new value:
w = z + 2u8;

If there are futures available to be executed, they will be listed by
numerical index, and you may run them using `#future` (or `#f`); for instance
#future 0
";

fn parse_breakpoint(s: &str) -> Option<Breakpoint> {
    let strings: Vec<&str> = s.split_whitespace().collect();
    if strings.len() == 2 {
        let mut program = strings[0].to_string();
        if program.ends_with(".aleo") {
            program.truncate(program.len() - 5);
        }
        if let Ok(line) = strings[1].parse::<usize>() {
            return Some(Breakpoint { program, line });
        }
    }
    None
}

/// Load all the Leo source files indicated and open the interpreter
/// to commands from the user.
pub fn interpret(filenames: &[PathBuf]) -> Result<()> {
    let mut interpreter = Interpreter::new(filenames.iter())?;
    let mut buffer = String::new();
    println!("{}", INSTRUCTIONS);
    loop {
        buffer.clear();
        if let Some(v) = interpreter.view_current_in_context() {
            println!("{v}");
        } else if let Some(v) = interpreter.view_current() {
            println!("{v}");
        }
        for (i, future) in interpreter.cursor.futures.iter().enumerate() {
            println!("{i}: {future}");
        }
        std::io::stdin().read_line(&mut buffer).expect("read_line");
        let action = match buffer.trim() {
            "#i" | "#into" => InterpreterAction::Into,
            "#s" | "#step" => InterpreterAction::Step,
            "#o" | "#over" => InterpreterAction::Over,
            "#r" | "#run" => InterpreterAction::Run,
            "#q" | "#quit" => return Ok(()),
            s => {
                if let Some(rest) = s.strip_prefix("#future ").or(s.strip_prefix("#f ")) {
                    if let Ok(num) = rest.trim().parse::<usize>() {
                        if num >= interpreter.cursor.futures.len() {
                            println!("No such future");
                            continue;
                        }
                        InterpreterAction::RunFuture(num)
                    } else {
                        println!("Failed to parse future");
                        continue;
                    }
                } else if let Some(rest) = s.strip_prefix("#break ").or(s.strip_prefix("#b ")) {
                    let Some(breakpoint) = parse_breakpoint(rest) else {
                        println!("Failed to parse breakpoint");
                        continue;
                    };
                    InterpreterAction::Breakpoint(breakpoint)
                } else if let Some(rest) = s.strip_prefix("#into ").or(s.strip_prefix("#i ")) {
                    InterpreterAction::LeoInterpretInto(rest.trim().into())
                } else {
                    InterpreterAction::LeoInterpretOver(s.trim().into())
                }
            }
        };

        match interpreter.action(action) {
            Ok(Some(value)) => println!("result: {value}"),
            Ok(None) => {}
            Err(LeoError::InterpreterHalt(interpreter_halt)) => println!("Halted: {interpreter_halt}"),
            Err(e) => return Err(e),
        }
    }
}
