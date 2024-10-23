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

use std::{collections::HashMap, fmt::Display, fs, path::Path};

use colored::*;

use snarkvm::prelude::TestnetV0;

use leo_ast::{Ast, Node as _, NodeBuilder};

use leo_passes::{Pass as _, SymbolTableCreator, TypeChecker, TypeTable};

use leo_span::{Span, source_map::FileName, symbol::with_session_globals};

use leo_errors::{CompilerError, InterpreterHalt, LeoError, Result, emitter::Handler};

mod cursor;
use cursor::*;

pub struct Interpreter {
    cursor: Cursor<'static>,
    cursor_initial: Cursor<'static>,
    actions: Vec<InterpreterAction>,
    handler: Handler,
    node_builder: NodeBuilder,
    breakpoints: Vec<Breakpoint>,
}

#[derive(Clone, Debug)]
pub struct Breakpoint {
    program: String,
    line: usize,
}

#[derive(Clone, Debug)]
pub enum InterpreterAction {
    LeoInterpretInto(String),
    LeoInterpretOver(String),
    Into,
    Over,
    Step,
    Breakpoint(Breakpoint),
    Run,
}

impl Interpreter {
    pub fn new<'a, P: 'a + AsRef<Path>>(source_files: impl IntoIterator<Item = &'a P>) -> Result<Self> {
        Self::new_impl(&mut source_files.into_iter().map(|p| p.as_ref()))
    }

    fn get_ast(path: &Path, handler: &Handler, node_builder: &NodeBuilder) -> Result<Ast> {
        let text = fs::read_to_string(path).map_err(|e| CompilerError::file_read_error(&path, e))?;
        let filename = FileName::Real(path.to_path_buf());
        let source_file = with_session_globals(|s| s.source_map.new_source(&text, filename));
        leo_parser::parse_ast::<TestnetV0>(handler, node_builder, &text, source_file.start_pos)
    }

    fn new_impl(source_files: &mut dyn Iterator<Item = &Path>) -> Result<Self> {
        let handler = Handler::default();
        let node_builder = Default::default();
        let mut cursor: Cursor<'_> = Cursor::default();
        for path in source_files {
            let ast = Self::get_ast(path, &handler, &node_builder)?;
            let symbol_table = SymbolTableCreator::do_pass((&ast, &handler))?;
            let type_table = TypeTable::default();
            TypeChecker::<TestnetV0>::do_pass((
                &ast,
                &handler,
                symbol_table,
                &type_table,
                10,    // conditional_block_max_depth
                false, // disable_conditional_branch_type_checking
            ))?;
            // TODO: This leak is silly.
            let ast = Box::leak(Box::new(ast));
            for (&program, scope) in ast.ast.program_scopes.iter() {
                for (name, function) in scope.functions.iter() {
                    cursor.functions.insert(GlobalId { program, name: *name }, function);
                }

                for (name, composite) in scope.structs.iter() {
                    cursor.structs.insert(
                        GlobalId { program, name: *name },
                        composite.members.iter().map(|member| member.identifier.name).collect(),
                    );
                }

                for (name, _mapping) in scope.structs.iter() {
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

        let cursor_initial = cursor.clone();

        Ok(Interpreter { cursor, cursor_initial, handler, node_builder, actions: Vec::new(), breakpoints: Vec::new() })
    }

    fn action(&mut self, act: InterpreterAction) -> Result<Option<Value>> {
        use InterpreterAction::*;

        let ret = match &act {
            LeoInterpretInto(s) | LeoInterpretOver(s) => {
                let s = s.trim();
                if s.ends_with(';') {
                    let statement = leo_parser::parse_statement::<TestnetV0>(&self.handler, &self.node_builder, s)
                        .map_err(|_e| {
                            LeoError::InterpreterHalt(InterpreterHalt::new("failed to parse statement".into()))
                        })?;
                    // TOOD: This leak is silly.
                    let stmt = Box::leak(Box::new(statement));
                    self.cursor.frames.push(Frame { step: 0, element: Element::Statement(stmt), user_initiated: true });
                } else {
                    let expression = leo_parser::parse_expression::<TestnetV0>(&self.handler, &self.node_builder, s)
                        .map_err(|_e| {
                            LeoError::InterpreterHalt(InterpreterHalt::new("failed to parse expression".into()))
                        })?;
                    // TOOD: This leak is silly.
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
                    self.cursor.step()?;
                }
                StepResult { finished: false, value: None }
            }
        };

        self.actions.push(act);

        Ok(ret.value)
    }

    pub fn view_current(&self) -> Option<impl Display> {
        if let Some(span) = self.current_span() {
            if span != Default::default() {
                return with_session_globals(|s| s.source_map.contents_of_span(span));
            }
        }

        Some(match self.cursor.frames.last()?.element {
            Element::Statement(statement) => format!("{statement}"),
            Element::Expression(expression) => format!("{expression}"),
            Element::Block { block, .. } => format!("{block}"),
            Element::Empty => tc_fail!(),
        })
    }

    pub fn view_current_in_context(&self) -> Option<impl Display> {
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
You may also use one letter abbreviations for these commands, such as #i.
Finally, you may simply enter expressions or statements on the command line
to evaluate. For instance, if you want to see the value of a variable w:
w
If you want to set w to a new value:
w = z + 2u8;
";

pub fn interpret(filenames: &[String]) -> Result<()> {
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
        std::io::stdin().read_line(&mut buffer).expect("read_line");
        let action = match buffer.trim() {
            "#i" | "#into" => InterpreterAction::Into,
            "#s" | "#step" => InterpreterAction::Step,
            "#o" | "#over" => InterpreterAction::Over,
            "#r" | "#run" => InterpreterAction::Run,
            "#q" | "#quit" => return Ok(()),
            s => {
                if let Some(rest) = s.strip_prefix("#into ").or(s.strip_prefix("#i ")) {
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
