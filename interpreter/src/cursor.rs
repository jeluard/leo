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

use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    mem,
    str::FromStr as _,
};

use indexmap::{IndexMap, IndexSet};

use rand::Rng as _;

use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

use snarkvm::prelude::{
    Address as SvmAddressParam,
    Boolean as SvmBooleanParam,
    Cast,
    CastLossy as _,
    Double as _,
    Field as SvmFieldParam,
    FromBits as _,
    Group as SvmGroupParam,
    Identifier as SvmIdentifierParam,
    Inverse as _,
    Network as _,
    Pow as _,
    Scalar as SvmScalarParam,
    // Signature as SvmSignatureParam,
    SizeInBits,
    Square as _,
    SquareRoot as _,
    TestnetV0,
    ToBits,
    integers::Integer as SvmIntegerParam,
};

type SvmAddress = SvmAddressParam<TestnetV0>;
type SvmBoolean = SvmBooleanParam<TestnetV0>;
type SvmField = SvmFieldParam<TestnetV0>;
type SvmGroup = SvmGroupParam<TestnetV0>;
type SvmIdentifier = SvmIdentifierParam<TestnetV0>;
type SvmInteger<I> = SvmIntegerParam<TestnetV0, I>;
type SvmScalar = SvmScalarParam<TestnetV0>;
// type SvmSignature = SvmSignatureParam<TestnetV0>;

use leo_ast::{
    AccessExpression,
    AssertVariant,
    BinaryOperation,
    Block,
    CoreConstant,
    CoreFunction,
    Expression,
    Function,
    GroupLiteral,
    IntegerType,
    Literal,
    Node as _,
    Statement,
    Type,
    UnaryOperation,
    Variant,
};

use leo_span::{Span, Symbol};

use leo_errors::{InterpreterHalt, Result};

#[macro_export]
macro_rules! tc_fail {
    () => {
        panic!("type checker failure")
    };
}

macro_rules! halt {
    ($span: expr) => {
        return Err(InterpreterHalt::new_spanned(String::new(), $span).into())

    };

    ($span: expr, $($x:tt)*) => {
        return Err(InterpreterHalt::new_spanned(format!($($x)*), $span).into())
    };
}

trait ExpectTc {
    type T;
    fn expect_tc(self, span: Span) -> Result<Self::T>;
}

impl<T> ExpectTc for Option<T> {
    type T = T;

    fn expect_tc(self, span: Span) -> Result<Self::T> {
        match self {
            Some(t) => Ok(t),
            None => Err(InterpreterHalt::new_spanned("type failure".into(), span).into()),
        }
    }
}

impl<T, U: std::fmt::Debug> ExpectTc for Result<T, U> {
    type T = T;

    fn expect_tc(self, span: Span) -> Result<Self::T> {
        self.map_err(|_e| InterpreterHalt::new_spanned("type failure".into(), span).into())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructContents {
    pub name: Symbol,
    pub contents: IndexMap<Symbol, Value>,
}

impl Hash for StructContents {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        for (_symbol, value) in self.contents.iter() {
            value.hash(state);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AsyncExecution {
    pub function: GlobalId,
    pub arguments: Vec<Value>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct Future(pub Vec<AsyncExecution>);

impl fmt::Display for Future {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Future")?;
        if !self.0.is_empty() {
            write!(f, " with calls to ")?;
            let mut names = self.0.iter().map(|async_ex| async_ex.function).peekable();
            while let Some(name) = names.next() {
                write!(f, "{name}")?;
                if names.peek().is_some() {
                    write!(f, ", ")?;
                }
            }
        }
        Ok(())
    }
}

/// A Leo value of any type.
///
/// Mappings and functions aren't considered values.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum Value {
    #[default]
    Unit,
    Bool(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    Group(SvmGroup),
    Field(SvmField),
    Scalar(SvmScalar),
    Array(Vec<Value>),
    // Signature(Box<SvmSignature>),
    Tuple(Vec<Value>),
    Address(SvmAddress),
    Future(Future),
    Struct(StructContents),
    // String(()),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Value::*;
        match self {
            Unit => write!(f, "()"),

            Bool(x) => write!(f, "{x}"),
            U8(x) => write!(f, "{x}u8"),
            U16(x) => write!(f, "{x}u16"),
            U32(x) => write!(f, "{x}u32"),
            U64(x) => write!(f, "{x}u64"),
            U128(x) => write!(f, "{x}u128"),
            I8(x) => write!(f, "{x}i8"),
            I16(x) => write!(f, "{x}i16"),
            I32(x) => write!(f, "{x}i32"),
            I64(x) => write!(f, "{x}i64"),
            I128(x) => write!(f, "{x}i128"),
            Group(x) => write!(f, "{x}"),
            Field(x) => write!(f, "{x}"),
            Scalar(x) => write!(f, "{x}"),
            Array(x) => {
                write!(f, "[")?;
                let mut iter = x.iter().peekable();
                while let Some(value) = iter.next() {
                    write!(f, "{value}")?;
                    if iter.peek().is_some() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")
            }
            Struct(StructContents { name, contents }) => {
                write!(f, "{name} {{")?;
                let mut iter = contents.iter().peekable();
                while let Some((member_name, value)) = iter.next() {
                    write!(f, "{member_name}: {value}")?;
                    if iter.peek().is_some() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "}}")
            }
            Tuple(x) => {
                write!(f, "(")?;
                let mut iter = x.iter().peekable();
                while let Some(value) = iter.next() {
                    write!(f, "{value}")?;
                    if iter.peek().is_some() {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")
            }
            Address(x) => write!(f, "{x}"),
            Future(future) => write!(f, "{future}"),
            // Signature(x) => write!(f, "{x}"),
            // String(_) => todo!(),
        }
    }
}

impl ToBits for Value {
    fn write_bits_le(&self, vec: &mut Vec<bool>) {
        use Value::*;

        macro_rules! literal {
            ($variant: expr, $x: expr) => {
                vec.push(false);
                vec.push(false);
                $variant.write_bits_le(vec);
                self.size_in_bits().write_bits_le(vec);
                $x.write_bits_le(vec);
            };
        }

        match self {
            // Literals
            Bool(x) => {
                literal!(1u8, x);
            }
            U8(x) => {
                literal!(9u8, x);
            }
            U16(x) => {
                literal!(10u8, x);
            }
            U32(x) => {
                literal!(11u8, x);
            }
            U64(x) => {
                literal!(12u8, x);
            }
            U128(x) => {
                literal!(13u8, x);
            }
            I8(x) => {
                literal!(4u8, x);
            }
            I16(x) => {
                literal!(5u8, x);
            }
            I32(x) => {
                literal!(6u8, x);
            }
            I64(x) => {
                literal!(7u8, x);
            }
            I128(x) => {
                literal!(8u8, x);
            }
            Group(x) => {
                literal!(3u8, x);
            }
            Field(x) => {
                literal!(2u8, x);
            }
            Scalar(x) => {
                literal!(14u8, x);
            }
            // Signature(x) => {
            //     literal!(15u8, x);
            // }
            Address(x) => {
                literal!(0u8, x);
            }

            Struct(StructContents { name: _, contents }) => {
                (contents.len() as u8).write_bits_le(vec);
                for (name, value) in contents.iter() {
                    let name_s = name.to_string();
                    let identifier = SvmIdentifier::from_str(&name_s).expect("identifier should parse");
                    identifier.size_in_bits().write_bits_le(vec);
                    identifier.write_bits_le(vec);
                    let value_bits = value.to_bits_le();
                    (value_bits.len() as u16).write_bits_le(vec);
                    vec.extend_from_slice(&value_bits);
                }
            }

            Array(array) => {
                for element in array.iter() {
                    let bits = element.to_bits_le();
                    (bits.len() as u16).write_bits_le(vec);
                    vec.extend_from_slice(&bits);
                }
            }

            Future(_) | Tuple(_) | Unit => tc_fail!(),
            // String(_) => {
            //     literal!(16u8);
            // }
        }
    }

    fn write_bits_be(&self, _vec: &mut Vec<bool>) {
        todo!()
    }
}

impl Value {
    fn to_fields(&self) -> Vec<SvmField> {
        let mut bits = self.to_bits_le();
        bits.push(true);
        bits.chunks(SvmField::SIZE_IN_DATA_BITS)
            .map(|bits| SvmField::from_bits_le(bits).expect("conversion should work"))
            .collect()
    }

    fn size_in_bits(&self) -> u16 {
        use Value::*;
        match self {
            Bool(_) => 1,
            U8(_) => 8,
            U16(_) => 16,
            U32(_) => 32,
            U64(_) => 64,
            U128(_) => 128,
            I8(_) => 8,
            I16(_) => 16,
            I32(_) => 32,
            I64(_) => 64,
            I128(_) => 128,
            Group(_) => SvmGroup::size_in_bits() as u16,
            Field(_) => SvmField::size_in_bits() as u16,
            Scalar(_) => SvmScalar::size_in_bits() as u16,
            // Signature(_) => SvmSignature::size_in_bits() as u16,
            Address(_) => SvmAddress::size_in_bits() as u16,

            Struct(StructContents { .. }) => todo!(),
            Array(_) => todo!(),
            Future(_) | Tuple(_) | Unit => tc_fail!(),
            // String(_) => todo!(),
        }
    }

    fn gte(&self, rhs: &Self) -> bool {
        rhs.lt(self)
    }

    fn lte(&self, rhs: &Self) -> bool {
        rhs.gt(self)
    }

    fn lt(&self, rhs: &Self) -> bool {
        use Value::*;
        match (self, rhs) {
            (U8(x), U8(y)) => x < y,
            (U16(x), U16(y)) => x < y,
            (U32(x), U32(y)) => x < y,
            (U64(x), U64(y)) => x < y,
            (U128(x), U128(y)) => x < y,
            (I8(x), I8(y)) => x < y,
            (I16(x), I16(y)) => x < y,
            (I32(x), I32(y)) => x < y,
            (I64(x), I64(y)) => x < y,
            (I128(x), I128(y)) => x < y,
            (Field(x), Field(y)) => x < y,
            _ => tc_fail!(),
        }
    }

    fn gt(&self, rhs: &Self) -> bool {
        use Value::*;
        match (self, rhs) {
            (U8(x), U8(y)) => x > y,
            (U16(x), U16(y)) => x > y,
            (U32(x), U32(y)) => x > y,
            (U64(x), U64(y)) => x > y,
            (U128(x), U128(y)) => x > y,
            (I8(x), I8(y)) => x > y,
            (I16(x), I16(y)) => x > y,
            (I32(x), I32(y)) => x > y,
            (I64(x), I64(y)) => x > y,
            (I128(x), I128(y)) => x > y,
            (Field(x), Field(y)) => x > y,
            _ => tc_fail!(),
        }
    }

    fn neq(&self, rhs: &Self) -> bool {
        !self.eq(rhs)
    }

    fn eq(&self, rhs: &Self) -> bool {
        use Value::*;
        match (self, rhs) {
            (Unit, Unit) => true,
            (Bool(x), Bool(y)) => x == y,
            (U8(x), U8(y)) => x == y,
            (U16(x), U16(y)) => x == y,
            (U32(x), U32(y)) => x == y,
            (U64(x), U64(y)) => x == y,
            (U128(x), U128(y)) => x == y,
            (I8(x), I8(y)) => x == y,
            (I16(x), I16(y)) => x == y,
            (I32(x), I32(y)) => x == y,
            (I64(x), I64(y)) => x == y,
            (I128(x), I128(y)) => x == y,
            (Field(x), Field(y)) => x == y,
            (Group(x), Group(y)) => x == y,
            (Array(x), Array(y)) => x.len() == y.len() && x.iter().zip(y.iter()).all(|(lhs, rhs)| lhs.eq(rhs)),
            _ => tc_fail!(),
        }
    }

    fn inc_unchecked(&self) -> Self {
        match self {
            Value::U8(x) => Value::U8(x.wrapping_add(1)),
            Value::U16(x) => Value::U16(x.wrapping_add(1)),
            Value::U32(x) => Value::U32(x.wrapping_add(1)),
            Value::U64(x) => Value::U64(x.wrapping_add(1)),
            Value::U128(x) => Value::U128(x.wrapping_add(1)),
            Value::I8(x) => Value::I8(x.wrapping_add(1)),
            Value::I16(x) => Value::I16(x.wrapping_add(1)),
            Value::I32(x) => Value::I32(x.wrapping_add(1)),
            Value::I64(x) => Value::I64(x.wrapping_add(1)),
            Value::I128(x) => Value::I128(x.wrapping_add(1)),
            _ => tc_fail!(),
        }
    }

    /// Doesn't correspond to Aleo's shl, because it
    /// does not fail when set bits are shifted out.
    fn simple_shl(&self, shift: u32) -> Self {
        match self {
            Value::U8(x) => Value::U8(x << shift),
            Value::U16(x) => Value::U16(x << shift),
            Value::U32(x) => Value::U32(x << shift),
            Value::U64(x) => Value::U64(x << shift),
            Value::U128(x) => Value::U128(x << shift),
            Value::I8(x) => Value::I8(x << shift),
            Value::I16(x) => Value::I16(x << shift),
            Value::I32(x) => Value::I32(x << shift),
            Value::I64(x) => Value::I64(x << shift),
            Value::I128(x) => Value::I128(x << shift),
            _ => tc_fail!(),
        }
    }

    fn simple_shr(&self, shift: u32) -> Self {
        match self {
            Value::U8(x) => Value::U8(x >> shift),
            Value::U16(x) => Value::U16(x >> shift),
            Value::U32(x) => Value::U32(x >> shift),
            Value::U64(x) => Value::U64(x >> shift),
            Value::U128(x) => Value::U128(x >> shift),
            Value::I8(x) => Value::I8(x >> shift),
            Value::I16(x) => Value::I16(x >> shift),
            Value::I32(x) => Value::I32(x >> shift),
            Value::I64(x) => Value::I64(x >> shift),
            Value::I128(x) => Value::I128(x >> shift),
            _ => tc_fail!(),
        }
    }
}

/// Names associated to values in a function being executed.
#[derive(Clone, Debug)]
struct FunctionContext {
    program: Symbol,
    names: HashMap<Symbol, Value>,
    accumulated_futures: Future,
    is_async: bool,
}

/// A stack of contexts, building with the function call stack.
#[derive(Clone, Debug, Default)]
pub struct ContextStack {
    contexts: Vec<FunctionContext>,
    current_len: usize,
}

impl ContextStack {
    fn len(&self) -> usize {
        self.current_len
    }

    fn push(&mut self, program: Symbol, is_async: bool) {
        if self.current_len == self.contexts.len() {
            self.contexts.push(FunctionContext {
                program,
                names: HashMap::new(),
                accumulated_futures: Default::default(),
                is_async,
            });
        }
        self.contexts[self.current_len].accumulated_futures.0.clear();
        self.contexts[self.current_len].names.clear();
        self.contexts[self.current_len].program = program;
        self.contexts[self.current_len].is_async = is_async;
        self.current_len += 1;
    }

    fn pop(&mut self) {
        // We never actually pop the underlying Vec
        // so we can reuse the storage of the hash
        // tables.
        assert!(self.len() > 0);
        self.current_len -= 1;
        self.contexts[self.current_len].names.clear();
    }

    /// Get the future accumulated by awaiting futures in the current function call.
    ///
    /// If the current code being interpreted is not in an async function, this
    /// will of course be empty.
    fn get_future(&mut self) -> Future {
        assert!(self.len() > 0);
        mem::take(&mut self.contexts[self.current_len - 1].accumulated_futures)
    }

    fn set(&mut self, symbol: Symbol, value: Value) {
        assert!(self.current_len > 0);
        self.last_mut().unwrap().names.insert(symbol, value);
    }

    fn add_future(&mut self, future: Future) {
        assert!(self.current_len > 0);
        self.contexts[self.current_len - 1].accumulated_futures.0.extend(future.0);
    }

    /// Are we currently in an async function?
    fn is_async(&self) -> bool {
        assert!(self.current_len > 0);
        self.last().unwrap().is_async
    }

    fn current_program(&self) -> Option<Symbol> {
        self.last().map(|c| c.program)
    }

    fn last(&self) -> Option<&FunctionContext> {
        self.len().checked_sub(1).and_then(|i| self.contexts.get(i))
    }

    fn last_mut(&mut self) -> Option<&mut FunctionContext> {
        self.len().checked_sub(1).and_then(|i| self.contexts.get_mut(i))
    }
}

/// A Leo construct to be evauated.
#[derive(Copy, Clone, Debug)]
pub enum Element<'a> {
    /// A Leo statement.
    Statement(&'a Statement),

    /// A Leo expression.
    Expression(&'a Expression),

    /// A Leo block.
    ///
    /// We have a separate variant for Leo blocks for two reasons:
    /// 1. In a ConditionalExpression, the `then` block is stored
    ///    as just a Block with no statement, and
    /// 2. We need to remember if a Block came from a function body,
    ///    so that if such a block ends, we know to push a `Unit` to
    ///    the values stack.
    Block {
        block: &'a Block,
        function_body: bool,
    },

    DelayedCall(GlobalId),
}

impl Element<'_> {
    pub fn span(&self) -> Span {
        use Element::*;
        match self {
            Statement(statement) => statement.span(),
            Expression(expression) => expression.span(),
            Block { block, .. } => block.span(),
            DelayedCall(..) => Default::default(),
        }
    }
}

/// A frame of execution, keeping track of the Element next to
/// be executed and the number of steps we've done so far.
#[derive(Clone, Debug)]
pub struct Frame<'a> {
    pub step: usize,
    pub element: Element<'a>,
    pub user_initiated: bool,
}

/// Global values - such as mappings, functions, etc -
/// are identified by program and name.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct GlobalId {
    pub program: Symbol,
    pub name: Symbol,
}

impl fmt::Display for GlobalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.program, self.name)
    }
}

/// Tracks the current execution state - a cursor into the running program.
#[derive(Clone, Debug)]
pub struct Cursor<'a> {
    /// Stack of execution frames, with the one currently to be executed on top.
    pub frames: Vec<Frame<'a>>,

    /// Stack of values from evaluated expressions.
    ///
    /// Each time an expression completes evaluation, a value is pushed here.
    pub values: Vec<Value>,

    /// All functions (or transitions or inlines) in any program being interpreted.
    pub functions: HashMap<GlobalId, &'a Function>,

    /// Consts are stored here.
    pub globals: HashMap<GlobalId, Value>,

    pub mappings: HashMap<GlobalId, HashMap<Value, Value>>,

    /// For each struct type, we only need to remember the names of its members, in order.
    pub structs: HashMap<GlobalId, IndexSet<Symbol>>,

    pub futures: Vec<Future>,

    pub contexts: ContextStack,

    rng: ChaCha20Rng,

    really_async: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(really_async: bool) -> Self {
        Cursor {
            frames: Default::default(),
            values: Default::default(),
            functions: Default::default(),
            globals: Default::default(),
            mappings: Default::default(),
            structs: Default::default(),
            contexts: Default::default(),
            futures: Default::default(),
            rng: ChaCha20Rng::from_entropy(),
            really_async,
        }
    }

    fn pop_value(&mut self) -> Result<Value> {
        match self.values.pop() {
            Some(v) => Ok(v),
            None => {
                Err(InterpreterHalt::new("value expected - this may be a bug in the Leo interpreter".to_string())
                    .into())
            }
        }
    }

    fn lookup(&self, name: Symbol) -> Option<Value> {
        let Some(context) = self.contexts.last() else {
            panic!("lookup requires a context");
        };

        let option_value = context.names.get(&name);
        if option_value.is_some() {
            return option_value.cloned();
        }

        self.globals.get(&GlobalId { program: context.program, name }).cloned()
    }

    fn lookup_mapping(&self, program: Option<Symbol>, name: Symbol) -> Option<&HashMap<Value, Value>> {
        let Some(program) = program.or(self.contexts.last().map(|c| c.program)) else {
            panic!("no program for mapping lookup");
        };
        self.mappings.get(&GlobalId { program, name })
    }

    fn lookup_mapping_mut(&mut self, program: Option<Symbol>, name: Symbol) -> Option<&mut HashMap<Value, Value>> {
        let Some(program) = program.or(self.contexts.last().map(|c| c.program)) else {
            panic!("no program for mapping lookup");
        };
        self.mappings.get_mut(&GlobalId { program, name })
    }

    fn lookup_function(&self, program: Symbol, name: Symbol) -> Option<&'a Function> {
        self.functions.get(&GlobalId { program, name }).cloned()
    }

    /// Execute the whole step of the current Element.
    ///
    /// That is, perform a step, and then finish all statements and expressions that have been pushed,
    /// until we're ready for the next step of the current Element (if there is one).
    pub fn whole_step(&mut self) -> Result<StepResult> {
        let frames_len = self.frames.len();
        let initial_result = self.step()?;
        if !initial_result.finished {
            while self.frames.len() > frames_len {
                self.step()?;
            }
        }
        Ok(initial_result)
    }

    /// Step `over` the current Element.
    ///
    /// That is, continue executing until the current Element is finished.
    pub fn over(&mut self) -> Result<StepResult> {
        let frames_len = self.frames.len();
        loop {
            match self.frames.len().cmp(&frames_len) {
                Ordering::Greater => {
                    self.step()?;
                }
                Ordering::Equal => {
                    let result = self.step()?;
                    if result.finished {
                        return Ok(result);
                    }
                }
                Ordering::Less => {
                    // This can happen if, for instance, a `return` was encountered,
                    // which means we exited the function we were evaluating and the
                    // frame stack was truncated.
                    return Ok(StepResult { finished: true, value: None });
                }
            }
        }
    }

    pub fn step_block(&mut self, block: &'a Block, function_body: bool, step: usize) -> bool {
        let len = self.frames.len();

        let done = match step {
            0 => {
                for statement in block.statements.iter().rev() {
                    self.frames.push(Frame { element: Element::Statement(statement), step: 0, user_initiated: false });
                }
                false
            }
            1 if function_body => {
                if self.contexts.is_async() {
                    let future = self.contexts.get_future();
                    self.values.push(Value::Future(future));
                } else {
                    self.values.push(Value::Unit);
                }
                self.contexts.pop();
                true
            }
            1 => true,
            _ => unreachable!(),
        };

        if done {
            assert_eq!(len, self.frames.len());
            self.frames.pop();
        } else {
            self.frames[len - 1].step += 1;
        }

        done
    }

    fn step_statement(&mut self, statement: &'a Statement, step: usize) -> Result<bool> {
        let len = self.frames.len();

        let mut push = |expression| {
            self.frames.push(Frame { element: Element::Expression(expression), step: 0, user_initiated: false })
        };

        let done = match statement {
            Statement::Assert(assert) if step == 0 => {
                match &assert.variant {
                    AssertVariant::Assert(x) => push(x),
                    AssertVariant::AssertEq(x, y) | AssertVariant::AssertNeq(x, y) => {
                        push(y);
                        push(x);
                    }
                };
                false
            }
            Statement::Assert(assert) if step == 1 => {
                match &assert.variant {
                    AssertVariant::Assert(..) => {
                        let value = self.pop_value()?;
                        match value {
                            Value::Bool(true) => {}
                            Value::Bool(false) => halt!(assert.span(), "assert failure"),
                            _ => tc_fail!(),
                        }
                    }
                    AssertVariant::AssertEq(..) | AssertVariant::AssertNeq(..) => {
                        let x = self.pop_value()?;
                        let y = self.pop_value()?;
                        let b =
                            if matches!(assert.variant, AssertVariant::AssertEq(..)) { x.eq(&y) } else { x.neq(&y) };
                        if !b {
                            halt!(assert.span(), "assert failure");
                        }
                    }
                };
                true
            }
            Statement::Assign(assign) if step == 0 => {
                push(&assign.value);
                false
            }
            Statement::Assign(assign) if step == 1 => {
                let value = self.values.pop().unwrap();
                let Expression::Identifier(id) = &assign.place else { tc_fail!() };
                self.contexts.set(id.name, value);
                true
            }
            Statement::Block(block) => return Ok(self.step_block(block, false, step)),
            Statement::Conditional(conditional) if step == 0 => {
                push(&conditional.condition);
                false
            }
            Statement::Conditional(conditional) if step == 1 => {
                match self.pop_value()? {
                    Value::Bool(true) => self.frames.push(Frame {
                        step: 0,
                        element: Element::Block { block: &conditional.then, function_body: false },
                        user_initiated: false,
                    }),
                    Value::Bool(false) if conditional.otherwise.is_some() => self.frames.push(Frame {
                        step: 0,
                        element: Element::Statement(conditional.otherwise.as_ref().unwrap()),
                        user_initiated: false,
                    }),
                    _ => tc_fail!(),
                };
                false
            }
            Statement::Conditional(_) if step == 2 => true,
            Statement::Console(_) => todo!(),
            Statement::Const(const_) if step == 0 => {
                push(&const_.value);
                false
            }
            Statement::Const(const_) if step == 1 => {
                let value = self.pop_value()?;
                self.contexts.set(const_.place.name, value);
                true
            }
            Statement::Definition(definition) if step == 0 => {
                push(&definition.value);
                false
            }
            Statement::Definition(definition) if step == 1 => {
                let Expression::Identifier(id) = &definition.place else {
                    tc_fail!();
                };
                let value = self.pop_value()?;
                self.contexts.set(id.name, value);
                true
            }
            Statement::Expression(expression) if step == 0 => {
                push(&expression.expression);
                false
            }
            Statement::Expression(_) if step == 1 => {
                self.values.pop();
                true
            }
            Statement::Iteration(iteration) if step == 0 => {
                assert!(!iteration.inclusive);
                push(&iteration.stop);
                push(&iteration.start);
                false
            }
            Statement::Iteration(iteration) => {
                // Currently there actually isn't a syntax in Leo for inclusive ranges.
                let stop = self.pop_value()?;
                let start = self.pop_value()?;
                if start.eq(&stop) {
                    true
                } else {
                    let new_start = start.inc_unchecked();
                    self.contexts.set(iteration.variable.name, start);
                    self.frames.push(Frame {
                        step: 0,
                        element: Element::Block { block: &iteration.block, function_body: false },
                        user_initiated: false,
                    });
                    self.values.push(new_start);
                    self.values.push(stop);
                    false
                }
            }
            Statement::Return(return_) if step == 0 => {
                push(&return_.expression);
                false
            }
            Statement::Return(_) if step == 1 => loop {
                let last_frame = self.frames.last().expect("a frame should be present");
                match last_frame.element {
                    Element::Expression(Expression::Call(_)) | Element::DelayedCall(_) => {
                        if self.contexts.is_async() {
                            // Get rid of the Unit we previously pushed, and replace it with a Future.
                            self.values.pop();
                            self.values.push(Value::Future(self.contexts.get_future()));
                        }
                        self.contexts.pop();
                        return Ok(true);
                    }
                    _ => {
                        self.frames.pop();
                    }
                }
            },
            _ => unreachable!(),
        };

        if done {
            assert_eq!(len, self.frames.len());
            self.frames.pop();
        } else {
            self.frames[len - 1].step += 1;
        }

        Ok(done)
    }

    fn step_expression(&mut self, expression: &'a Expression, step: usize) -> Result<bool> {
        let len = self.frames.len();

        macro_rules! push {
            () => {
                |expression| {
                    self.frames.push(Frame { element: Element::Expression(expression), step: 0, user_initiated: false })
                }
            };
        }

        if let Some(value) = match expression {
            Expression::Access(AccessExpression::Array(array)) if step == 0 => {
                push!()(&*array.index);
                push!()(&*array.array);
                None
            }
            Expression::Access(AccessExpression::Array(array)) if step == 1 => {
                let span = array.span();
                let array = self.pop_value()?;
                let index = self.pop_value()?;

                let index_usize: usize = match index {
                    Value::U8(x) => x.into(),
                    Value::U16(x) => x.into(),
                    Value::U32(x) => x.try_into().expect_tc(span)?,
                    Value::U64(x) => x.try_into().expect_tc(span)?,
                    Value::U128(x) => x.try_into().expect_tc(span)?,
                    Value::I8(x) => x.try_into().expect_tc(span)?,
                    Value::I16(x) => x.try_into().expect_tc(span)?,
                    Value::I32(x) => x.try_into().expect_tc(span)?,
                    Value::I64(x) => x.try_into().expect_tc(span)?,
                    Value::I128(x) => x.try_into().expect_tc(span)?,
                    _ => tc_fail!(),
                };
                let Value::Array(vec_array) = array else { tc_fail!() };
                Some(vec_array.get(index_usize).expect_tc(span)?.clone())
            }
            Expression::Access(AccessExpression::AssociatedConstant(constant)) if step == 0 => {
                let Type::Identifier(type_ident) = constant.ty else {
                    tc_fail!();
                };
                let Some(core_constant) = CoreConstant::from_symbols(type_ident.name, constant.name.name) else {
                    tc_fail!();
                };
                match core_constant {
                    CoreConstant::GroupGenerator => Some(Value::Group(SvmGroup::generator())),
                }
            }
            Expression::Access(AccessExpression::AssociatedFunction(function)) if step == 0 => {
                let Some(core_function) = CoreFunction::from_symbols(function.variant.name, function.name.name) else {
                    tc_fail!();
                };

                // We want to push expressions for each of the arguments... except for mappings,
                // because we don't look them up as Values.
                match core_function {
                    CoreFunction::MappingGet | CoreFunction::MappingRemove | CoreFunction::MappingContains => {
                        push!()(&function.arguments[1]);
                    }
                    CoreFunction::MappingGetOrUse | CoreFunction::MappingSet => {
                        push!()(&function.arguments[2]);
                        push!()(&function.arguments[1]);
                    }
                    _ => function.arguments.iter().rev().for_each(push!()),
                }
                None
            }
            Expression::Access(AccessExpression::AssociatedFunction(function)) if step == 1 => {
                let Some(core_function) = CoreFunction::from_symbols(function.variant.name, function.name.name) else {
                    tc_fail!();
                };

                let span = function.span();

                macro_rules! apply {
                    ($func: expr, $value: ident, $to: ident) => {{
                        let v = self.pop_value()?;
                        let bits = v.$to();
                        Value::$value($func(&bits).expect_tc(span)?)
                    }};
                }

                macro_rules! apply_cast {
                    ($func: expr, $value: ident, $to: ident) => {{
                        let v = self.pop_value()?;
                        let bits = v.$to();
                        let group = $func(&bits).expect_tc(span)?;
                        let x = group.to_x_coordinate();
                        Value::$value(x.cast_lossy())
                    }};
                }

                macro_rules! apply_cast_int {
                    ($func: expr, $value: ident, $int_ty: ident, $to: ident) => {{
                        let v = self.pop_value()?;
                        let bits = v.$to();
                        let group = $func(&bits).expect_tc(span)?;
                        let x = group.to_x_coordinate();
                        let bits = x.to_bits_le();
                        let mut result: $int_ty = 0;
                        for bit in 0..std::cmp::min($int_ty::BITS as usize, bits.len()) {
                            let setbit = (if bits[bit] { 1 } else { 0 }) << bit;
                            result |= setbit;
                        }
                        Value::$value(result)
                    }};
                }

                macro_rules! apply_cast2 {
                    ($func: expr, $value: ident) => {{
                        let Value::Scalar(randomizer) = self.pop_value()? else {
                            tc_fail!();
                        };
                        let v = self.pop_value()?;
                        let bits = v.to_bits_le();
                        let group = $func(&bits, &randomizer).expect_tc(span)?;
                        let x = group.to_x_coordinate();
                        Value::$value(x.cast_lossy())
                    }};
                }

                let value = match core_function {
                    CoreFunction::BHP256CommitToAddress => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp256, Address)
                    }
                    CoreFunction::BHP256CommitToField => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp256, Field)
                    }
                    CoreFunction::BHP256CommitToGroup => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp256, Group)
                    }
                    CoreFunction::BHP256HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_bhp256, Address, to_bits_le)
                    }
                    CoreFunction::BHP256HashToField => apply!(TestnetV0::hash_bhp256, Field, to_bits_le),
                    CoreFunction::BHP256HashToGroup => apply!(TestnetV0::hash_to_group_bhp256, Group, to_bits_le),
                    CoreFunction::BHP256HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, I8, i8, to_bits_le)
                    }
                    CoreFunction::BHP256HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, I16, i16, to_bits_le)
                    }
                    CoreFunction::BHP256HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, I32, i32, to_bits_le)
                    }
                    CoreFunction::BHP256HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, I64, i64, to_bits_le)
                    }
                    CoreFunction::BHP256HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, I128, i128, to_bits_le)
                    }
                    CoreFunction::BHP256HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, U8, u8, to_bits_le)
                    }
                    CoreFunction::BHP256HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, U16, u16, to_bits_le)
                    }
                    CoreFunction::BHP256HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, U32, u32, to_bits_le)
                    }
                    CoreFunction::BHP256HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, U64, u64, to_bits_le)
                    }
                    CoreFunction::BHP256HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp256, U128, u128, to_bits_le)
                    }
                    CoreFunction::BHP256HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_bhp256, Scalar, to_bits_le)
                    }
                    CoreFunction::BHP512CommitToAddress => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp512, Address)
                    }
                    CoreFunction::BHP512CommitToField => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp512, Field)
                    }
                    CoreFunction::BHP512CommitToGroup => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp512, Group)
                    }
                    CoreFunction::BHP512HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_bhp512, Address, to_bits_le)
                    }
                    CoreFunction::BHP512HashToField => apply!(TestnetV0::hash_bhp512, Field, to_bits_le),
                    CoreFunction::BHP512HashToGroup => apply!(TestnetV0::hash_to_group_bhp512, Group, to_bits_le),
                    CoreFunction::BHP512HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, I8, i8, to_bits_le)
                    }
                    CoreFunction::BHP512HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, I16, i16, to_bits_le)
                    }
                    CoreFunction::BHP512HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, I32, i32, to_bits_le)
                    }
                    CoreFunction::BHP512HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, I64, i64, to_bits_le)
                    }
                    CoreFunction::BHP512HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, I128, i128, to_bits_le)
                    }
                    CoreFunction::BHP512HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, U8, u8, to_bits_le)
                    }
                    CoreFunction::BHP512HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, U16, u16, to_bits_le)
                    }
                    CoreFunction::BHP512HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, U32, u32, to_bits_le)
                    }
                    CoreFunction::BHP512HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, U64, u64, to_bits_le)
                    }
                    CoreFunction::BHP512HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp512, U128, u128, to_bits_le)
                    }
                    CoreFunction::BHP512HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_bhp512, Scalar, to_bits_le)
                    }
                    CoreFunction::BHP768CommitToAddress => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp768, Address)
                    }
                    CoreFunction::BHP768CommitToField => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp768, Field)
                    }
                    CoreFunction::BHP768CommitToGroup => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp768, Group)
                    }
                    CoreFunction::BHP768HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_bhp768, Address, to_bits_le)
                    }
                    CoreFunction::BHP768HashToField => apply!(TestnetV0::hash_bhp768, Field, to_bits_le),
                    CoreFunction::BHP768HashToGroup => apply!(TestnetV0::hash_to_group_bhp768, Group, to_bits_le),
                    CoreFunction::BHP768HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, I8, i8, to_bits_le)
                    }
                    CoreFunction::BHP768HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, I16, i16, to_bits_le)
                    }
                    CoreFunction::BHP768HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, I32, i32, to_bits_le)
                    }
                    CoreFunction::BHP768HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, I64, i64, to_bits_le)
                    }
                    CoreFunction::BHP768HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, I128, i128, to_bits_le)
                    }
                    CoreFunction::BHP768HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, U8, u8, to_bits_le)
                    }
                    CoreFunction::BHP768HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, U16, u16, to_bits_le)
                    }
                    CoreFunction::BHP768HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, U32, u32, to_bits_le)
                    }
                    CoreFunction::BHP768HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, U64, u64, to_bits_le)
                    }
                    CoreFunction::BHP768HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp768, U128, u128, to_bits_le)
                    }
                    CoreFunction::BHP768HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_bhp768, Scalar, to_bits_le)
                    }
                    CoreFunction::BHP1024CommitToAddress => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp1024, Address)
                    }
                    CoreFunction::BHP1024CommitToField => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp1024, Field)
                    }
                    CoreFunction::BHP1024CommitToGroup => {
                        apply_cast2!(TestnetV0::commit_to_group_bhp1024, Group)
                    }
                    CoreFunction::BHP1024HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_bhp1024, Address, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToField => apply!(TestnetV0::hash_bhp1024, Field, to_bits_le),
                    CoreFunction::BHP1024HashToGroup => apply!(TestnetV0::hash_to_group_bhp1024, Group, to_bits_le),
                    CoreFunction::BHP1024HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, I8, i8, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, I16, i16, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, I32, i32, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, I64, i64, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, I128, i128, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, U8, u8, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, U16, u16, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, U32, u32, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, U64, u64, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_bhp1024, U128, u128, to_bits_le)
                    }
                    CoreFunction::BHP1024HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_bhp1024, Scalar, to_bits_le)
                    }
                    CoreFunction::ChaChaRandAddress => Value::Address(self.rng.gen()),
                    CoreFunction::ChaChaRandBool => Value::Bool(self.rng.gen()),
                    CoreFunction::ChaChaRandField => Value::Field(self.rng.gen()),
                    CoreFunction::ChaChaRandGroup => Value::Group(self.rng.gen()),
                    CoreFunction::ChaChaRandI8 => Value::I8(self.rng.gen()),
                    CoreFunction::ChaChaRandI16 => Value::I16(self.rng.gen()),
                    CoreFunction::ChaChaRandI32 => Value::I32(self.rng.gen()),
                    CoreFunction::ChaChaRandI64 => Value::I64(self.rng.gen()),
                    CoreFunction::ChaChaRandI128 => Value::I128(self.rng.gen()),
                    CoreFunction::ChaChaRandU8 => Value::U8(self.rng.gen()),
                    CoreFunction::ChaChaRandU16 => Value::U16(self.rng.gen()),
                    CoreFunction::ChaChaRandU32 => Value::U32(self.rng.gen()),
                    CoreFunction::ChaChaRandU64 => Value::U64(self.rng.gen()),
                    CoreFunction::ChaChaRandU128 => Value::U128(self.rng.gen()),
                    CoreFunction::ChaChaRandScalar => Value::Scalar(self.rng.gen()),
                    CoreFunction::Keccak256HashToAddress => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        Address,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToField => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        Field,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToGroup => {
                        apply!(
                            |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                            Group,
                            to_bits_le
                        )
                    }
                    CoreFunction::Keccak256HashToI8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        I8,
                        i8,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToI16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        I16,
                        i16,
                        to_bits_le
                    ),

                    CoreFunction::Keccak256HashToI32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        I32,
                        i32,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToI64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        I64,
                        i64,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToI128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        I128,
                        i128,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToU8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        U8,
                        u8,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToU16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        U16,
                        u16,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToU32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        U32,
                        u32,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToU64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        U64,
                        u64,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToU128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        U128,
                        u128,
                        to_bits_le
                    ),
                    CoreFunction::Keccak256HashToScalar => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_keccak256(v).expect_tc(span)?),
                        Scalar,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToAddress => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        Address,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToField => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        Field,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToGroup => {
                        apply!(
                            |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                            Group,
                            to_bits_le
                        )
                    }
                    CoreFunction::Keccak384HashToI8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        I8,
                        i8,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToI16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        I16,
                        i16,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToI32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        I32,
                        i32,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToI64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        I64,
                        i64,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToI128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        I128,
                        i128,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToU8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        U8,
                        u8,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToU16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        U16,
                        u16,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToU32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        U32,
                        u32,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToU64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        U64,
                        u64,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToU128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        U128,
                        u128,
                        to_bits_le
                    ),
                    CoreFunction::Keccak384HashToScalar => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak384(v).expect_tc(span)?),
                        Scalar,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToAddress => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        Address,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToField => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        Field,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToGroup => {
                        apply!(
                            |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                            Group,
                            to_bits_le
                        )
                    }
                    CoreFunction::Keccak512HashToI8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        I8,
                        i8,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToI16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        I16,
                        i16,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToI32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        I32,
                        i32,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToI64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        I64,
                        i64,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToI128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        I128,
                        i128,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToU8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        U8,
                        u8,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToU16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        U16,
                        u16,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToU32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        U32,
                        u32,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToU64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        U64,
                        u64,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToU128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        U128,
                        u128,
                        to_bits_le
                    ),
                    CoreFunction::Keccak512HashToScalar => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_keccak512(v).expect_tc(span)?),
                        Scalar,
                        to_bits_le
                    ),
                    CoreFunction::Pedersen64CommitToAddress => {
                        apply_cast2!(TestnetV0::commit_to_group_ped64, Address)
                    }
                    CoreFunction::Pedersen64CommitToField => {
                        apply_cast2!(TestnetV0::commit_to_group_ped64, Field)
                    }
                    CoreFunction::Pedersen64CommitToGroup => {
                        apply_cast2!(TestnetV0::commit_to_group_ped64, Group)
                    }
                    CoreFunction::Pedersen64HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_ped64, Address, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToField => apply!(TestnetV0::hash_ped64, Field, to_bits_le),
                    CoreFunction::Pedersen64HashToGroup => apply!(TestnetV0::hash_to_group_ped64, Group, to_bits_le),
                    CoreFunction::Pedersen64HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I8, i8, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I16, i16, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I32, i32, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I64, i64, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I128, i128, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U8, u8, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U16, u16, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U32, u32, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U64, u64, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U128, u128, to_bits_le)
                    }
                    CoreFunction::Pedersen64HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_ped64, Scalar, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_ped128, Address, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToField => {
                        apply_cast!(TestnetV0::hash_to_group_ped128, Field, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToGroup => {
                        apply_cast!(TestnetV0::hash_to_group_ped128, Group, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped128, I8, i8, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I16, i16, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped128, I32, i32, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, I64, i64, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped128, I128, i128, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped128, U8, u8, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U16, u16, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped128, U32, u32, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped64, U64, u64, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_ped128, U128, u128, to_bits_le)
                    }
                    CoreFunction::Pedersen128HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_ped128, Scalar, to_bits_le)
                    }
                    CoreFunction::Pedersen128CommitToAddress => {
                        apply_cast2!(TestnetV0::commit_to_group_ped128, Address)
                    }
                    CoreFunction::Pedersen128CommitToField => {
                        apply_cast2!(TestnetV0::commit_to_group_ped128, Field)
                    }
                    CoreFunction::Pedersen128CommitToGroup => {
                        apply_cast2!(TestnetV0::commit_to_group_ped128, Group)
                    }
                    CoreFunction::Poseidon2HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_psd2, Address, to_fields)
                    }
                    CoreFunction::Poseidon2HashToField => {
                        apply_cast!(TestnetV0::hash_to_group_psd2, Field, to_fields)
                    }
                    CoreFunction::Poseidon2HashToGroup => {
                        apply_cast!(TestnetV0::hash_to_group_psd2, Group, to_fields)
                    }
                    CoreFunction::Poseidon2HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, I8, i8, to_fields)
                    }
                    CoreFunction::Poseidon2HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, I16, i16, to_fields)
                    }
                    CoreFunction::Poseidon2HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, I32, i32, to_fields)
                    }
                    CoreFunction::Poseidon2HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, I64, i64, to_fields)
                    }
                    CoreFunction::Poseidon2HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, I128, i128, to_fields)
                    }
                    CoreFunction::Poseidon2HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, U8, u8, to_fields)
                    }
                    CoreFunction::Poseidon2HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, U16, u16, to_fields)
                    }
                    CoreFunction::Poseidon2HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, U32, u32, to_fields)
                    }
                    CoreFunction::Poseidon2HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, U64, u64, to_fields)
                    }
                    CoreFunction::Poseidon2HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd2, U128, u128, to_fields)
                    }
                    CoreFunction::Poseidon2HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_psd4, Scalar, to_fields)
                    }
                    CoreFunction::Poseidon4HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_psd4, Address, to_fields)
                    }
                    CoreFunction::Poseidon4HashToField => {
                        apply_cast!(TestnetV0::hash_to_group_psd4, Field, to_fields)
                    }
                    CoreFunction::Poseidon4HashToGroup => {
                        apply_cast!(TestnetV0::hash_to_group_psd4, Group, to_fields)
                    }
                    CoreFunction::Poseidon4HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, I8, i8, to_fields)
                    }
                    CoreFunction::Poseidon4HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, I16, i16, to_fields)
                    }
                    CoreFunction::Poseidon4HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, I32, i32, to_fields)
                    }
                    CoreFunction::Poseidon4HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, I64, i64, to_fields)
                    }
                    CoreFunction::Poseidon4HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, I128, i128, to_fields)
                    }
                    CoreFunction::Poseidon4HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, U8, u8, to_fields)
                    }
                    CoreFunction::Poseidon4HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, U16, u16, to_fields)
                    }
                    CoreFunction::Poseidon4HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, U32, u32, to_fields)
                    }
                    CoreFunction::Poseidon4HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, U64, u64, to_fields)
                    }
                    CoreFunction::Poseidon4HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd4, U128, u128, to_fields)
                    }
                    CoreFunction::Poseidon4HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_psd4, Scalar, to_fields)
                    }
                    CoreFunction::Poseidon8HashToAddress => {
                        apply_cast!(TestnetV0::hash_to_group_psd8, Address, to_fields)
                    }
                    CoreFunction::Poseidon8HashToField => {
                        apply_cast!(TestnetV0::hash_to_group_psd8, Field, to_fields)
                    }
                    CoreFunction::Poseidon8HashToGroup => {
                        apply_cast!(TestnetV0::hash_to_group_psd8, Group, to_fields)
                    }
                    CoreFunction::Poseidon8HashToI8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, I8, i8, to_fields)
                    }
                    CoreFunction::Poseidon8HashToI16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, I16, i16, to_fields)
                    }
                    CoreFunction::Poseidon8HashToI32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, I32, i32, to_fields)
                    }
                    CoreFunction::Poseidon8HashToI64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, I64, i64, to_fields)
                    }
                    CoreFunction::Poseidon8HashToI128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, I128, i128, to_fields)
                    }
                    CoreFunction::Poseidon8HashToU8 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, U8, u8, to_fields)
                    }
                    CoreFunction::Poseidon8HashToU16 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, U16, u16, to_fields)
                    }
                    CoreFunction::Poseidon8HashToU32 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, U32, u32, to_fields)
                    }
                    CoreFunction::Poseidon8HashToU64 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, U64, u64, to_fields)
                    }
                    CoreFunction::Poseidon8HashToU128 => {
                        apply_cast_int!(TestnetV0::hash_to_group_psd8, U128, u128, to_fields)
                    }
                    CoreFunction::Poseidon8HashToScalar => {
                        apply_cast!(TestnetV0::hash_to_group_psd8, Scalar, to_fields)
                    }
                    CoreFunction::SHA3_256HashToAddress => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        Address,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToField => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        Field,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToGroup => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        Group,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToI8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        I8,
                        i8,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToI16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        I16,
                        i16,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToI32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        I32,
                        i32,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToI64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        I64,
                        i64,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToI128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        I128,
                        i128,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToU8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        U8,
                        u8,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToU16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        U16,
                        u16,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToU32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        U32,
                        u32,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToU64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        U64,
                        u64,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToU128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        U128,
                        u128,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_256HashToScalar => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp256(&TestnetV0::hash_sha3_256(v).expect_tc(span)?),
                        Scalar,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToAddress => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        Address,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToField => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        Field,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToGroup => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        Group,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToI8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        I8,
                        i8,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToI16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        I16,
                        i16,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToI32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        I32,
                        i32,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToI64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        I64,
                        i64,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToI128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        I128,
                        i128,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToU8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        U8,
                        u8,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToU16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        U16,
                        u16,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToU32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        U32,
                        u32,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToU64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        U64,
                        u64,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToU128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        U128,
                        u128,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_384HashToScalar => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_384(v).expect_tc(span)?),
                        Scalar,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToAddress => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        Address,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToField => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        Field,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToGroup => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        Group,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToI8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        I8,
                        i8,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToI16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        I16,
                        i16,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToI32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        I32,
                        i32,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToI64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        I64,
                        i64,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToI128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        I128,
                        i128,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToU8 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        U8,
                        u8,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToU16 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        U16,
                        u16,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToU32 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        U32,
                        u32,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToU64 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        U64,
                        u64,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToU128 => apply_cast_int!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        U128,
                        u128,
                        to_bits_le
                    ),
                    CoreFunction::SHA3_512HashToScalar => apply_cast!(
                        |v| TestnetV0::hash_to_group_bhp512(&TestnetV0::hash_sha3_512(v).expect_tc(span)?),
                        Scalar,
                        to_bits_le
                    ),
                    CoreFunction::MappingGet => {
                        let key = self.values.pop().expect_tc(span)?;
                        let (program, name) = match &function.arguments[0] {
                            Expression::Identifier(id) => (None, id.name),
                            Expression::Locator(locator) => (Some(locator.program.name.name), locator.name),
                            _ => tc_fail!(),
                        };
                        match self.lookup_mapping(program, name).and_then(|mapping| mapping.get(&key)) {
                            Some(v) => v.clone(),
                            None => halt!(function.span(), "map lookup failure"),
                        }
                    }
                    CoreFunction::MappingGetOrUse => {
                        let use_value = self.values.pop().expect_tc(span)?;
                        let key = self.values.pop().expect_tc(span)?;
                        let (program, name) = match &function.arguments[0] {
                            Expression::Identifier(id) => (None, id.name),
                            Expression::Locator(locator) => (Some(locator.program.name.name), locator.name),
                            _ => tc_fail!(),
                        };
                        match self.lookup_mapping(program, name).and_then(|mapping| mapping.get(&key)) {
                            Some(v) => v.clone(),
                            None => use_value,
                        }
                    }
                    CoreFunction::MappingSet => {
                        let value = self.pop_value()?;
                        let key = self.pop_value()?;
                        let (program, name) = match &function.arguments[0] {
                            Expression::Identifier(id) => (None, id.name),
                            Expression::Locator(locator) => (Some(locator.program.name.name), locator.name),
                            _ => tc_fail!(),
                        };
                        if let Some(mapping) = self.lookup_mapping_mut(program, name) {
                            mapping.insert(key, value);
                        } else {
                            tc_fail!();
                        }
                        Value::Unit
                    }
                    CoreFunction::MappingRemove => {
                        let key = self.pop_value()?;
                        let (program, name) = match &function.arguments[0] {
                            Expression::Identifier(id) => (None, id.name),
                            Expression::Locator(locator) => (Some(locator.program.name.name), locator.name),
                            _ => tc_fail!(),
                        };
                        if let Some(mapping) = self.lookup_mapping_mut(program, name) {
                            mapping.remove(&key);
                        } else {
                            tc_fail!();
                        }
                        Value::Unit
                    }
                    CoreFunction::MappingContains => {
                        let key = self.pop_value()?;
                        let (program, name) = match &function.arguments[0] {
                            Expression::Identifier(id) => (None, id.name),
                            Expression::Locator(locator) => (Some(locator.program.name.name), locator.name),
                            _ => tc_fail!(),
                        };
                        if let Some(mapping) = self.lookup_mapping_mut(program, name) {
                            Value::Bool(mapping.contains_key(&key))
                        } else {
                            tc_fail!();
                        }
                    }
                    CoreFunction::GroupToXCoordinate => {
                        let Value::Group(g) = self.pop_value()? else {
                            tc_fail!();
                        };
                        Value::Field(g.to_x_coordinate())
                    }
                    CoreFunction::GroupToYCoordinate => {
                        let Value::Group(g) = self.pop_value()? else {
                            tc_fail!();
                        };
                        Value::Field(g.to_y_coordinate())
                    }
                    CoreFunction::SignatureVerify => todo!(),
                    CoreFunction::FutureAwait => {
                        let Value::Future(future) = self.pop_value()? else {
                            tc_fail!();
                        };
                        self.contexts.add_future(future);
                        Value::Unit
                    }
                };
                Some(value)
            }
            Expression::Array(array) if step == 0 => {
                array.elements.iter().rev().for_each(push!());
                None
            }
            Expression::Array(array) if step == 1 => {
                let len = self.values.len();
                let array_values = self.values.drain(len - array.elements.len()..).collect();
                Some(Value::Array(array_values))
            }
            Expression::Binary(binary) if step == 0 => {
                push!()(&binary.right);
                push!()(&binary.left);
                None
            }
            Expression::Binary(binary) if step == 1 => {
                let rhs = self.pop_value()?;
                let lhs = self.pop_value()?;
                Some(evaluate_binary(binary.span, binary.op, lhs, rhs)?)
            }
            Expression::Call(call) if step == 0 => {
                call.arguments.iter().rev().for_each(push!());
                None
            }
            Expression::Call(call) if step == 1 => {
                let len = self.values.len();
                let (program, name) = match &*call.function {
                    Expression::Identifier(id) => {
                        let program = call.program.unwrap_or_else(|| {
                            self.contexts.current_program().expect("there should be a current program")
                        });
                        (program, id.name)
                    }
                    Expression::Locator(locator) => (locator.program.name.name, locator.name),
                    _ => tc_fail!(),
                };
                let function = self.lookup_function(program, name).expect_tc(call.span())?;
                let values_iter = self.values.drain(len - call.arguments.len()..);
                if self.really_async && function.variant == Variant::AsyncFunction {
                    // Don't actually run the call now.
                    let async_ex =
                        AsyncExecution { function: GlobalId { name, program }, arguments: values_iter.collect() };
                    self.values.push(Value::Future(Future(vec![async_ex])));
                } else {
                    let is_async = function.variant == Variant::AsyncFunction;
                    self.contexts.push(program, is_async);
                    let param_names = function.input.iter().map(|input| input.identifier.name);
                    for (name, value) in param_names.zip(values_iter) {
                        self.contexts.set(name, value);
                    }
                    self.frames.push(Frame {
                        step: 0,
                        element: Element::Block { block: &function.block, function_body: true },
                        user_initiated: false,
                    });
                }
                None
            }
            Expression::Call(_call) if step == 2 => Some(self.pop_value()?),
            Expression::Cast(cast) if step == 0 => {
                push!()(&*cast.expression);
                None
            }
            Expression::Cast(cast) if step == 1 => {
                fn make_value<C>(span: Span, c: C, cast_type: &Type) -> Result<Value>
                where
                    C: Cast<SvmAddress>
                        + Cast<SvmField>
                        + Cast<SvmAddress>
                        + Cast<SvmGroup>
                        + Cast<SvmBoolean>
                        + Cast<SvmScalar>
                        + Cast<SvmInteger<u8>>
                        + Cast<SvmInteger<u16>>
                        + Cast<SvmInteger<u32>>
                        + Cast<SvmInteger<u64>>
                        + Cast<SvmInteger<u128>>
                        + Cast<SvmInteger<i8>>
                        + Cast<SvmInteger<i16>>
                        + Cast<SvmInteger<i32>>
                        + Cast<SvmInteger<i64>>
                        + Cast<SvmInteger<i128>>,
                {
                    use Type::*;

                    let fail = |_t| InterpreterHalt::new_spanned("cast failure".to_string(), span);

                    let value = match cast_type {
                        Address => Value::Address(c.cast().map_err(fail)?),
                        Boolean => Value::Bool({
                            let b: SvmBoolean = c.cast().map_err(fail)?;
                            *b
                        }),
                        Field => Value::Field(c.cast().map_err(fail)?),
                        Group => Value::Group(c.cast().map_err(fail)?),
                        Integer(IntegerType::U8) => Value::U8({
                            let i: SvmInteger<u8> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::U16) => Value::U16({
                            let i: SvmInteger<u16> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::U32) => Value::U32({
                            let i: SvmInteger<u32> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::U64) => Value::U64({
                            let i: SvmInteger<u64> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::U128) => Value::U128({
                            let i: SvmInteger<u128> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::I8) => Value::I8({
                            let i: SvmInteger<i8> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::I16) => Value::I16({
                            let i: SvmInteger<i16> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::I32) => Value::I32({
                            let i: SvmInteger<i32> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::I64) => Value::I64({
                            let i: SvmInteger<i64> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Integer(IntegerType::I128) => Value::I128({
                            let i: SvmInteger<i128> = c.cast().map_err(fail)?;
                            *i
                        }),
                        Scalar => Value::Scalar(c.cast().map_err(fail)?),

                        _ => tc_fail!(),
                    };
                    Ok(value)
                }

                let span = cast.span();
                let arg = self.pop_value()?;

                Some(match arg {
                    Value::Bool(b) => make_value(span, SvmBoolean::new(b), &cast.type_),
                    Value::U8(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::U16(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::U32(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::U64(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::U128(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::I8(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::I16(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::I32(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::I64(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::I128(x) => make_value(span, SvmInteger::new(x), &cast.type_),
                    Value::Group(g) => make_value(span, g.to_x_coordinate(), &cast.type_),
                    Value::Field(f) => make_value(span, f, &cast.type_),
                    Value::Scalar(s) => make_value(span, s, &cast.type_),
                    Value::Address(a) => make_value(span, a.to_group().to_x_coordinate(), &cast.type_),
                    _ => tc_fail!(),
                }?)
            }
            Expression::Err(_) => todo!(),
            Expression::Identifier(identifier) if step == 0 => {
                Some(self.lookup(identifier.name).expect_tc(identifier.span())?)
            }
            Expression::Literal(literal) if step == 0 => Some(match literal {
                Literal::Boolean(b, ..) => Value::Bool(*b),
                Literal::Integer(IntegerType::U8, s, ..) => Value::U8(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::U16, s, ..) => Value::U16(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::U32, s, ..) => Value::U32(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::U64, s, ..) => Value::U64(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::U128, s, ..) => Value::U128(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::I8, s, ..) => Value::I8(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::I16, s, ..) => Value::I16(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::I32, s, ..) => Value::I32(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::I64, s, ..) => Value::I64(s.parse().expect_tc(literal.span())?),
                Literal::Integer(IntegerType::I128, s, ..) => Value::I128(s.parse().expect_tc(literal.span())?),
                Literal::Field(s, ..) => Value::Field(format!("{s}field").parse().expect_tc(literal.span())?),
                Literal::Group(group_literal) => match &**group_literal {
                    GroupLiteral::Single(s, ..) => Value::Group(format!("{s}group").parse().expect_tc(literal.span())?),
                    GroupLiteral::Tuple(_group_tuple) => todo!(),
                },
                Literal::Address(s, ..) => Value::Address(s.parse().expect_tc(literal.span())?),
                Literal::Scalar(s, ..) => Value::Scalar(format!("{s}scalar").parse().expect_tc(literal.span())?),
                Literal::String(..) => tc_fail!(),
            }),
            Expression::Locator(_locator) => todo!(),
            Expression::Struct(struct_) if step == 0 => {
                struct_.members.iter().rev().flat_map(|init| init.expression.as_ref()).for_each(push!());
                None
            }
            Expression::Struct(struct_) if step == 1 => {
                // Collect all the key/value pairs into a HashMap.
                let mut contents_tmp = HashMap::with_capacity(struct_.members.len());
                for initializer in struct_.members.iter() {
                    let name = initializer.identifier.name;
                    let value = if initializer.expression.is_some() {
                        self.pop_value()?
                    } else {
                        self.lookup(name).expect_tc(struct_.span())?
                    };
                    contents_tmp.insert(name, value);
                }

                // And now put them into an IndexMap in the correct order.
                let program = self.contexts.current_program().expect("there should be a current program");
                let id = GlobalId { program, name: struct_.name.name };
                let struct_type = self.structs.get(&id).expect_tc(struct_.span())?;
                let contents = struct_type
                    .iter()
                    .map(|sym| (*sym, contents_tmp.remove(sym).expect("we just inserted this")))
                    .collect();

                Some(Value::Struct(StructContents { name: struct_.name.name, contents }))
            }
            Expression::Ternary(ternary) if step == 0 => {
                push!()(&*ternary.condition);
                None
            }
            Expression::Ternary(ternary) if step == 1 => {
                let condition = self.pop_value()?;
                match condition {
                    Value::Bool(true) => push!()(&*ternary.if_true),
                    Value::Bool(false) => push!()(&*ternary.if_false),
                    _ => tc_fail!(),
                }
                None
            }
            Expression::Ternary(_) if step == 2 => Some(self.pop_value()?),
            Expression::Tuple(tuple) if step == 0 => {
                tuple.elements.iter().rev().for_each(push!());
                None
            }
            Expression::Tuple(tuple) if step == 1 => {
                let len = self.values.len();
                let tuple_values = self.values.drain(len - tuple.elements.len()..).collect();
                Some(Value::Tuple(tuple_values))
            }
            Expression::Unary(unary) if step == 0 => {
                push!()(&*unary.receiver);
                None
            }
            Expression::Unary(unary) if step == 1 => {
                let value = self.pop_value()?;
                Some(evaluate_unary(unary.span, unary.op, value)?)
            }
            Expression::Unit(_) if step == 0 => Some(Value::Unit),
            _ => unreachable!(),
        } {
            assert_eq!(self.frames.len(), len);
            self.frames.pop();
            self.values.push(value);
            Ok(true)
        } else {
            self.frames[len - 1].step += 1;
            Ok(false)
        }
    }

    /// Execute one step of the current element.
    ///
    /// Many Leo constructs require multiple steps. For instance, when executing a conditional,
    /// the first step will push the condition expression to the stack. Once that has executed
    /// and we've returned to the conditional, we push the `then` or `otherwise` block to the
    /// stack. Once that has executed and we've returned to the conditional, the final step
    /// does nothing.
    pub fn step(&mut self) -> Result<StepResult> {
        if self.frames.is_empty() {
            return Err(InterpreterHalt::new("no execution frames available".into()).into());
        }

        let Frame { element, step, user_initiated } = self.frames.last().expect("there should be a frame");
        let user_initiated = *user_initiated;
        match element {
            Element::Block { block, function_body } => {
                let finished = self.step_block(block, *function_body, *step);
                Ok(StepResult { finished, value: None })
            }
            Element::Statement(statement) => {
                let finished = self.step_statement(statement, *step)?;
                Ok(StepResult { finished, value: None })
            }
            Element::Expression(expression) => {
                let finished = self.step_expression(expression, *step)?;
                let value = match (finished, user_initiated) {
                    (false, _) => None,
                    (true, false) => self.values.last().cloned(),
                    (true, true) => self.values.pop(),
                };
                if let Some(Value::Future(future)) = &value {
                    if user_initiated && !future.0.is_empty() {
                        self.futures.push(future.clone());
                    }
                }
                Ok(StepResult { finished, value })
            }
            Element::DelayedCall(gid) if *step == 0 => {
                let function = self.lookup_function(gid.program, gid.name).expect("function should exist");
                assert!(function.variant == Variant::AsyncFunction);
                let len = self.values.len();
                let values_iter = self.values.drain(len - function.input.len()..);
                self.contexts.push(
                    gid.program,
                    true, // is_async
                );
                let param_names = function.input.iter().map(|input| input.identifier.name);
                for (name, value) in param_names.zip(values_iter) {
                    self.contexts.set(name, value);
                }
                self.frames.last_mut().unwrap().step = 1;
                self.frames.push(Frame {
                    step: 0,
                    element: Element::Block { block: &function.block, function_body: true },
                    user_initiated: false,
                });
                Ok(StepResult { finished: false, value: None })
            }
            Element::DelayedCall(_gid) => {
                assert_eq!(*step, 1);
                let value = self.values.pop();
                let Some(Value::Future(future)) = value else {
                    panic!("Delayed calls should always be to async functions");
                };
                if !future.0.is_empty() {
                    self.futures.push(future.clone());
                }
                self.frames.pop();
                Ok(StepResult { finished: true, value: Some(Value::Future(future)) })
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct StepResult {
    /// Has this element completely finished running?
    pub finished: bool,

    /// If the element was an expression, here's its value.
    pub value: Option<Value>,
}

/// Evaluate a binary operation.
fn evaluate_binary(span: Span, op: BinaryOperation, lhs: Value, rhs: Value) -> Result<Value> {
    let value = match op {
        BinaryOperation::Add => {
            let Some(value) = (match (lhs, rhs) {
                (Value::U8(x), Value::U8(y)) => x.checked_add(y).map(Value::U8),
                (Value::U16(x), Value::U16(y)) => x.checked_add(y).map(Value::U16),
                (Value::U32(x), Value::U32(y)) => x.checked_add(y).map(Value::U32),
                (Value::U64(x), Value::U64(y)) => x.checked_add(y).map(Value::U64),
                (Value::U128(x), Value::U128(y)) => x.checked_add(y).map(Value::U128),
                (Value::I8(x), Value::I8(y)) => x.checked_add(y).map(Value::I8),
                (Value::I16(x), Value::I16(y)) => x.checked_add(y).map(Value::I16),
                (Value::I32(x), Value::I32(y)) => x.checked_add(y).map(Value::I32),
                (Value::I64(x), Value::I64(y)) => x.checked_add(y).map(Value::I64),
                (Value::I128(x), Value::I128(y)) => x.checked_add(y).map(Value::I128),
                (Value::Group(x), Value::Group(y)) => Some(Value::Group(x + y)),
                (Value::Field(x), Value::Field(y)) => Some(Value::Field(x + y)),
                (Value::Scalar(x), Value::Scalar(y)) => Some(Value::Scalar(x + y)),
                _ => tc_fail!(),
            }) else {
                halt!(span, "add overflow");
            };
            value
        }
        BinaryOperation::AddWrapped => match (lhs, rhs) {
            (Value::U8(x), Value::U8(y)) => Value::U8(x.wrapping_add(y)),
            (Value::U16(x), Value::U16(y)) => Value::U16(x.wrapping_add(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.wrapping_add(y)),
            (Value::U64(x), Value::U64(y)) => Value::U64(x.wrapping_add(y)),
            (Value::U128(x), Value::U128(y)) => Value::U128(x.wrapping_add(y)),
            (Value::I8(x), Value::I8(y)) => Value::I8(x.wrapping_add(y)),
            (Value::I16(x), Value::I16(y)) => Value::I16(x.wrapping_add(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.wrapping_add(y)),
            (Value::I64(x), Value::I64(y)) => Value::I64(x.wrapping_add(y)),
            (Value::I128(x), Value::I128(y)) => Value::I128(x.wrapping_add(y)),
            _ => tc_fail!(),
        },
        BinaryOperation::And => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x && y),
            _ => tc_fail!(),
        },
        BinaryOperation::BitwiseAnd => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x & y),
            (Value::U8(x), Value::U8(y)) => Value::U8(x & y),
            (Value::U16(x), Value::U16(y)) => Value::U16(x & y),
            (Value::U32(x), Value::U32(y)) => Value::U32(x & y),
            (Value::U64(x), Value::U64(y)) => Value::U64(x & y),
            (Value::U128(x), Value::U128(y)) => Value::U128(x & y),
            (Value::I8(x), Value::I8(y)) => Value::I8(x & y),
            (Value::I16(x), Value::I16(y)) => Value::I16(x & y),
            (Value::I32(x), Value::I32(y)) => Value::I32(x & y),
            (Value::I64(x), Value::I64(y)) => Value::I64(x & y),
            (Value::I128(x), Value::I128(y)) => Value::I128(x & y),
            _ => tc_fail!(),
        },
        BinaryOperation::Div => {
            let Some(value) = (match (lhs, rhs) {
                (Value::U8(x), Value::U8(y)) => x.checked_div(y).map(Value::U8),
                (Value::U16(x), Value::U16(y)) => x.checked_div(y).map(Value::U16),
                (Value::U32(x), Value::U32(y)) => x.checked_div(y).map(Value::U32),
                (Value::U64(x), Value::U64(y)) => x.checked_div(y).map(Value::U64),
                (Value::U128(x), Value::U128(y)) => x.checked_div(y).map(Value::U128),
                (Value::I8(x), Value::I8(y)) => x.checked_div(y).map(Value::I8),
                (Value::I16(x), Value::I16(y)) => x.checked_div(y).map(Value::I16),
                (Value::I32(x), Value::I32(y)) => x.checked_div(y).map(Value::I32),
                (Value::I64(x), Value::I64(y)) => x.checked_div(y).map(Value::I64),
                (Value::I128(x), Value::I128(y)) => x.checked_div(y).map(Value::I128),
                (Value::Field(x), Value::Field(y)) => y.inverse().map(|y| Value::Field(x * y)).ok(),
                _ => tc_fail!(),
            }) else {
                halt!(span, "div overflow");
            };
            value
        }
        BinaryOperation::DivWrapped => match (lhs, rhs) {
            (Value::U8(_), Value::U8(0))
            | (Value::U16(_), Value::U16(0))
            | (Value::U32(_), Value::U32(0))
            | (Value::U64(_), Value::U64(0))
            | (Value::U128(_), Value::U128(0))
            | (Value::I8(_), Value::I8(0))
            | (Value::I16(_), Value::I16(0))
            | (Value::I32(_), Value::I32(0))
            | (Value::I64(_), Value::I64(0))
            | (Value::I128(_), Value::I128(0)) => halt!(span, "divide by 0"),
            (Value::U8(x), Value::U8(y)) => Value::U8(x.wrapping_div(y)),
            (Value::U16(x), Value::U16(y)) => Value::U16(x.wrapping_div(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.wrapping_div(y)),
            (Value::U64(x), Value::U64(y)) => Value::U64(x.wrapping_div(y)),
            (Value::U128(x), Value::U128(y)) => Value::U128(x.wrapping_div(y)),
            (Value::I8(x), Value::I8(y)) => Value::I8(x.wrapping_div(y)),
            (Value::I16(x), Value::I16(y)) => Value::I16(x.wrapping_div(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.wrapping_div(y)),
            (Value::I64(x), Value::I64(y)) => Value::I64(x.wrapping_div(y)),
            (Value::I128(x), Value::I128(y)) => Value::I128(x.wrapping_div(y)),
            _ => tc_fail!(),
        },
        BinaryOperation::Eq => Value::Bool(lhs.eq(&rhs)),
        BinaryOperation::Gte => Value::Bool(lhs.gte(&rhs)),
        BinaryOperation::Gt => Value::Bool(lhs.gt(&rhs)),
        BinaryOperation::Lte => Value::Bool(lhs.lte(&rhs)),
        BinaryOperation::Lt => Value::Bool(lhs.lt(&rhs)),
        BinaryOperation::Mod => {
            let Some(value) = (match (lhs, rhs) {
                (Value::U8(x), Value::U8(y)) => x.checked_rem(y).map(Value::U8),
                (Value::U16(x), Value::U16(y)) => x.checked_rem(y).map(Value::U16),
                (Value::U32(x), Value::U32(y)) => x.checked_rem(y).map(Value::U32),
                (Value::U64(x), Value::U64(y)) => x.checked_rem(y).map(Value::U64),
                (Value::U128(x), Value::U128(y)) => x.checked_rem(y).map(Value::U128),
                (Value::I8(x), Value::I8(y)) => x.checked_rem(y).map(Value::I8),
                (Value::I16(x), Value::I16(y)) => x.checked_rem(y).map(Value::I16),
                (Value::I32(x), Value::I32(y)) => x.checked_rem(y).map(Value::I32),
                (Value::I64(x), Value::I64(y)) => x.checked_rem(y).map(Value::I64),
                (Value::I128(x), Value::I128(y)) => x.checked_rem(y).map(Value::I128),
                _ => tc_fail!(),
            }) else {
                halt!(span, "mod overflow");
            };
            value
        }
        BinaryOperation::Mul => {
            let Some(value) = (match (lhs, rhs) {
                (Value::U8(x), Value::U8(y)) => x.checked_mul(y).map(Value::U8),
                (Value::U16(x), Value::U16(y)) => x.checked_mul(y).map(Value::U16),
                (Value::U32(x), Value::U32(y)) => x.checked_mul(y).map(Value::U32),
                (Value::U64(x), Value::U64(y)) => x.checked_mul(y).map(Value::U64),
                (Value::U128(x), Value::U128(y)) => x.checked_mul(y).map(Value::U128),
                (Value::I8(x), Value::I8(y)) => x.checked_mul(y).map(Value::I8),
                (Value::I16(x), Value::I16(y)) => x.checked_mul(y).map(Value::I16),
                (Value::I32(x), Value::I32(y)) => x.checked_mul(y).map(Value::I32),
                (Value::I64(x), Value::I64(y)) => x.checked_mul(y).map(Value::I64),
                (Value::I128(x), Value::I128(y)) => x.checked_mul(y).map(Value::I128),
                (Value::Field(x), Value::Field(y)) => Some(Value::Field(x * y)),
                (Value::Group(x), Value::Scalar(y)) => Some(Value::Group(x * y)),
                (Value::Scalar(x), Value::Group(y)) => Some(Value::Group(x * y)),
                _ => tc_fail!(),
            }) else {
                halt!(span, "mul overflow");
            };
            value
        }
        BinaryOperation::MulWrapped => match (lhs, rhs) {
            (Value::U8(x), Value::U8(y)) => Value::U8(x.wrapping_mul(y)),
            (Value::U16(x), Value::U16(y)) => Value::U16(x.wrapping_mul(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.wrapping_mul(y)),
            (Value::U64(x), Value::U64(y)) => Value::U64(x.wrapping_mul(y)),
            (Value::U128(x), Value::U128(y)) => Value::U128(x.wrapping_mul(y)),
            (Value::I8(x), Value::I8(y)) => Value::I8(x.wrapping_mul(y)),
            (Value::I16(x), Value::I16(y)) => Value::I16(x.wrapping_mul(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.wrapping_mul(y)),
            (Value::I64(x), Value::I64(y)) => Value::I64(x.wrapping_mul(y)),
            (Value::I128(x), Value::I128(y)) => Value::I128(x.wrapping_mul(y)),
            (Value::Field(_), Value::Field(_)) => todo!(),
            _ => tc_fail!(),
        },

        BinaryOperation::Nand => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(!(x & y)),
            _ => tc_fail!(),
        },
        BinaryOperation::Neq => Value::Bool(lhs.neq(&rhs)),
        BinaryOperation::Nor => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(!(x | y)),
            _ => tc_fail!(),
        },
        BinaryOperation::Or => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x | y),
            _ => tc_fail!(),
        },
        BinaryOperation::BitwiseOr => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x | y),
            (Value::U8(x), Value::U8(y)) => Value::U8(x | y),
            (Value::U16(x), Value::U16(y)) => Value::U16(x | y),
            (Value::U32(x), Value::U32(y)) => Value::U32(x | y),
            (Value::U64(x), Value::U64(y)) => Value::U64(x | y),
            (Value::U128(x), Value::U128(y)) => Value::U128(x | y),
            (Value::I8(x), Value::I8(y)) => Value::I8(x | y),
            (Value::I16(x), Value::I16(y)) => Value::I16(x | y),
            (Value::I32(x), Value::I32(y)) => Value::I32(x | y),
            (Value::I64(x), Value::I64(y)) => Value::I64(x | y),
            (Value::I128(x), Value::I128(y)) => Value::I128(x | y),
            _ => tc_fail!(),
        },
        BinaryOperation::Pow => {
            if let (Value::Field(x), Value::Field(y)) = (&lhs, &rhs) {
                Value::Field(x.pow(y))
            } else {
                let rhs: u32 = match rhs {
                    Value::U8(y) => y.into(),
                    Value::U16(y) => y.into(),
                    Value::U32(y) => y,
                    _ => tc_fail!(),
                };

                let Some(value) = (match lhs {
                    Value::U8(x) => x.checked_pow(rhs).map(Value::U8),
                    Value::U16(x) => x.checked_pow(rhs).map(Value::U16),
                    Value::U32(x) => x.checked_pow(rhs).map(Value::U32),
                    Value::U64(x) => x.checked_pow(rhs).map(Value::U64),
                    Value::U128(x) => x.checked_pow(rhs).map(Value::U128),
                    Value::I8(x) => x.checked_pow(rhs).map(Value::I8),
                    Value::I16(x) => x.checked_pow(rhs).map(Value::I16),
                    Value::I32(x) => x.checked_pow(rhs).map(Value::I32),
                    Value::I64(x) => x.checked_pow(rhs).map(Value::I64),
                    Value::I128(x) => x.checked_pow(rhs).map(Value::I128),
                    _ => tc_fail!(),
                }) else {
                    halt!(span, "pow overflow");
                };
                value
            }
        }
        BinaryOperation::PowWrapped => {
            let rhs: u32 = match rhs {
                Value::U8(y) => y.into(),
                Value::U16(y) => y.into(),
                Value::U32(y) => y,
                _ => tc_fail!(),
            };

            match lhs {
                Value::U8(x) => Value::U8(x.wrapping_pow(rhs)),
                Value::U16(x) => Value::U16(x.wrapping_pow(rhs)),
                Value::U32(x) => Value::U32(x.wrapping_pow(rhs)),
                Value::U64(x) => Value::U64(x.wrapping_pow(rhs)),
                Value::U128(x) => Value::U128(x.wrapping_pow(rhs)),
                Value::I8(x) => Value::I8(x.wrapping_pow(rhs)),
                Value::I16(x) => Value::I16(x.wrapping_pow(rhs)),
                Value::I32(x) => Value::I32(x.wrapping_pow(rhs)),
                Value::I64(x) => Value::I64(x.wrapping_pow(rhs)),
                Value::I128(x) => Value::I128(x.wrapping_pow(rhs)),
                _ => tc_fail!(),
            }
        }
        BinaryOperation::Rem => {
            let Some(value) = (match (lhs, rhs) {
                (Value::U8(x), Value::U8(y)) => x.checked_rem(y).map(Value::U8),
                (Value::U16(x), Value::U16(y)) => x.checked_rem(y).map(Value::U16),
                (Value::U32(x), Value::U32(y)) => x.checked_rem(y).map(Value::U32),
                (Value::U64(x), Value::U64(y)) => x.checked_rem(y).map(Value::U64),
                (Value::U128(x), Value::U128(y)) => x.checked_rem(y).map(Value::U128),
                (Value::I8(x), Value::I8(y)) => x.checked_rem(y).map(Value::I8),
                (Value::I16(x), Value::I16(y)) => x.checked_rem(y).map(Value::I16),
                (Value::I32(x), Value::I32(y)) => x.checked_rem(y).map(Value::I32),
                (Value::I64(x), Value::I64(y)) => x.checked_rem(y).map(Value::I64),
                (Value::I128(x), Value::I128(y)) => x.checked_rem(y).map(Value::I128),
                _ => tc_fail!(),
            }) else {
                halt!(span, "rem error");
            };
            value
        }
        BinaryOperation::RemWrapped => match (lhs, rhs) {
            (Value::U8(_), Value::U8(0))
            | (Value::U16(_), Value::U16(0))
            | (Value::U32(_), Value::U32(0))
            | (Value::U64(_), Value::U64(0))
            | (Value::U128(_), Value::U128(0))
            | (Value::I8(_), Value::I8(0))
            | (Value::I16(_), Value::I16(0))
            | (Value::I32(_), Value::I32(0))
            | (Value::I64(_), Value::I64(0))
            | (Value::I128(_), Value::I128(0)) => halt!(span, "rem by 0"),
            (Value::U8(x), Value::U8(y)) => Value::U8(x.wrapping_rem(y)),
            (Value::U16(x), Value::U16(y)) => Value::U16(x.wrapping_rem(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.wrapping_rem(y)),
            (Value::U64(x), Value::U64(y)) => Value::U64(x.wrapping_rem(y)),
            (Value::U128(x), Value::U128(y)) => Value::U128(x.wrapping_rem(y)),
            (Value::I8(x), Value::I8(y)) => Value::I8(x.wrapping_rem(y)),
            (Value::I16(x), Value::I16(y)) => Value::I16(x.wrapping_rem(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.wrapping_rem(y)),
            (Value::I64(x), Value::I64(y)) => Value::I64(x.wrapping_rem(y)),
            (Value::I128(x), Value::I128(y)) => Value::I128(x.wrapping_rem(y)),
            _ => tc_fail!(),
        },
        BinaryOperation::Shl => {
            let rhs: u32 = match rhs {
                Value::U8(y) => y.into(),
                Value::U16(y) => y.into(),
                Value::U32(y) => y,
                _ => tc_fail!(),
            };
            match lhs {
                Value::U8(_) | Value::I8(_) if rhs >= 8 => halt!(span, "shl overflow"),
                Value::U16(_) | Value::I16(_) if rhs >= 16 => halt!(span, "shl overflow"),
                Value::U32(_) | Value::I32(_) if rhs >= 32 => halt!(span, "shl overflow"),
                Value::U64(_) | Value::I64(_) if rhs >= 64 => halt!(span, "shl overflow"),
                Value::U128(_) | Value::I128(_) if rhs >= 128 => halt!(span, "shl overflow"),
                _ => {}
            }

            // Aleo's shl halts if set bits are shifted out.
            let shifted = lhs.simple_shl(rhs);
            let reshifted = shifted.simple_shr(rhs);
            if lhs.eq(&reshifted) {
                shifted
            } else {
                halt!(span, "shl overflow");
            }
        }

        BinaryOperation::ShlWrapped => {
            let rhs: u32 = match rhs {
                Value::U8(y) => y.into(),
                Value::U16(y) => y.into(),
                Value::U32(y) => y,
                _ => tc_fail!(),
            };
            match lhs {
                Value::U8(x) => Value::U8(x.wrapping_shl(rhs)),
                Value::U16(x) => Value::U16(x.wrapping_shl(rhs)),
                Value::U32(x) => Value::U32(x.wrapping_shl(rhs)),
                Value::U64(x) => Value::U64(x.wrapping_shl(rhs)),
                Value::U128(x) => Value::U128(x.wrapping_shl(rhs)),
                Value::I8(x) => Value::I8(x.wrapping_shl(rhs)),
                Value::I16(x) => Value::I16(x.wrapping_shl(rhs)),
                Value::I32(x) => Value::I32(x.wrapping_shl(rhs)),
                Value::I64(x) => Value::I64(x.wrapping_shl(rhs)),
                Value::I128(x) => Value::I128(x.wrapping_shl(rhs)),
                _ => tc_fail!(),
            }
        }

        BinaryOperation::Shr => {
            let rhs: u32 = match rhs {
                Value::U8(y) => y.into(),
                Value::U16(y) => y.into(),
                Value::U32(y) => y,
                _ => tc_fail!(),
            };

            match lhs {
                Value::U8(_) | Value::I8(_) if rhs >= 8 => halt!(span, "shr overflow"),
                Value::U16(_) | Value::I16(_) if rhs >= 16 => halt!(span, "shr overflow"),
                Value::U32(_) | Value::I32(_) if rhs >= 32 => halt!(span, "shr overflow"),
                Value::U64(_) | Value::I64(_) if rhs >= 64 => halt!(span, "shr overflow"),
                Value::U128(_) | Value::I128(_) if rhs >= 128 => halt!(span, "shr overflow"),
                _ => {}
            }

            lhs.simple_shr(rhs)
        }

        BinaryOperation::ShrWrapped => {
            let rhs: u32 = match rhs {
                Value::U8(y) => y.into(),
                Value::U16(y) => y.into(),
                Value::U32(y) => y,
                _ => tc_fail!(),
            };

            match lhs {
                Value::U8(x) => Value::U8(x.wrapping_shr(rhs)),
                Value::U16(x) => Value::U16(x.wrapping_shr(rhs)),
                Value::U32(x) => Value::U32(x.wrapping_shr(rhs)),
                Value::U64(x) => Value::U64(x.wrapping_shr(rhs)),
                Value::U128(x) => Value::U128(x.wrapping_shr(rhs)),
                Value::I8(x) => Value::I8(x.wrapping_shr(rhs)),
                Value::I16(x) => Value::I16(x.wrapping_shr(rhs)),
                Value::I32(x) => Value::I32(x.wrapping_shr(rhs)),
                Value::I64(x) => Value::I64(x.wrapping_shr(rhs)),
                Value::I128(x) => Value::I128(x.wrapping_shr(rhs)),
                _ => tc_fail!(),
            }
        }

        BinaryOperation::Sub => {
            let Some(value) = (match (lhs, rhs) {
                (Value::U8(x), Value::U8(y)) => x.checked_sub(y).map(Value::U8),
                (Value::U16(x), Value::U16(y)) => x.checked_sub(y).map(Value::U16),
                (Value::U32(x), Value::U32(y)) => x.checked_sub(y).map(Value::U32),
                (Value::U64(x), Value::U64(y)) => x.checked_sub(y).map(Value::U64),
                (Value::U128(x), Value::U128(y)) => x.checked_sub(y).map(Value::U128),
                (Value::I8(x), Value::I8(y)) => x.checked_sub(y).map(Value::I8),
                (Value::I16(x), Value::I16(y)) => x.checked_sub(y).map(Value::I16),
                (Value::I32(x), Value::I32(y)) => x.checked_sub(y).map(Value::I32),
                (Value::I64(x), Value::I64(y)) => x.checked_sub(y).map(Value::I64),
                (Value::I128(x), Value::I128(y)) => x.checked_sub(y).map(Value::I128),
                (Value::Group(x), Value::Group(y)) => Some(Value::Group(x - y)),
                (Value::Field(x), Value::Field(y)) => Some(Value::Field(x - y)),
                _ => tc_fail!(),
            }) else {
                halt!(span, "sub overflow");
            };
            value
        }

        BinaryOperation::SubWrapped => match (lhs, rhs) {
            (Value::U8(x), Value::U8(y)) => Value::U8(x.wrapping_sub(y)),
            (Value::U16(x), Value::U16(y)) => Value::U16(x.wrapping_sub(y)),
            (Value::U32(x), Value::U32(y)) => Value::U32(x.wrapping_sub(y)),
            (Value::U64(x), Value::U64(y)) => Value::U64(x.wrapping_sub(y)),
            (Value::U128(x), Value::U128(y)) => Value::U128(x.wrapping_sub(y)),
            (Value::I8(x), Value::I8(y)) => Value::I8(x.wrapping_sub(y)),
            (Value::I16(x), Value::I16(y)) => Value::I16(x.wrapping_sub(y)),
            (Value::I32(x), Value::I32(y)) => Value::I32(x.wrapping_sub(y)),
            (Value::I64(x), Value::I64(y)) => Value::I64(x.wrapping_sub(y)),
            (Value::I128(x), Value::I128(y)) => Value::I128(x.wrapping_sub(y)),
            _ => tc_fail!(),
        },

        BinaryOperation::Xor => match (lhs, rhs) {
            (Value::Bool(x), Value::Bool(y)) => Value::Bool(x ^ y),
            (Value::U8(x), Value::U8(y)) => Value::U8(x ^ y),
            (Value::U16(x), Value::U16(y)) => Value::U16(x ^ y),
            (Value::U32(x), Value::U32(y)) => Value::U32(x ^ y),
            (Value::U64(x), Value::U64(y)) => Value::U64(x ^ y),
            (Value::U128(x), Value::U128(y)) => Value::U128(x ^ y),
            (Value::I8(x), Value::I8(y)) => Value::I8(x ^ y),
            (Value::I16(x), Value::I16(y)) => Value::I16(x ^ y),
            (Value::I32(x), Value::I32(y)) => Value::I32(x ^ y),
            (Value::I64(x), Value::I64(y)) => Value::I64(x ^ y),
            (Value::I128(x), Value::I128(y)) => Value::I128(x ^ y),
            _ => tc_fail!(),
        },
    };
    Ok(value)
}

/// Evaluate a unary operation.
fn evaluate_unary(span: Span, op: UnaryOperation, value: Value) -> Result<Value> {
    let value_result = match op {
        UnaryOperation::Abs => match value {
            Value::I8(x) => {
                if x == i8::MIN {
                    halt!(span, "abs overflow");
                } else {
                    Value::I8(x.abs())
                }
            }
            Value::I16(x) => {
                if x == i16::MIN {
                    halt!(span, "abs overflow");
                } else {
                    Value::I16(x.abs())
                }
            }
            Value::I32(x) => {
                if x == i32::MIN {
                    halt!(span, "abs overflow");
                } else {
                    Value::I32(x.abs())
                }
            }
            Value::I64(x) => {
                if x == i64::MIN {
                    halt!(span, "abs overflow");
                } else {
                    Value::I64(x.abs())
                }
            }
            Value::I128(x) => {
                if x == i128::MIN {
                    halt!(span, "abs overflow");
                } else {
                    Value::I128(x.abs())
                }
            }
            _ => tc_fail!(),
        },
        UnaryOperation::AbsWrapped => match value {
            Value::I8(x) => Value::I8(x.unsigned_abs() as i8),
            Value::I16(x) => Value::I16(x.unsigned_abs() as i16),
            Value::I32(x) => Value::I32(x.unsigned_abs() as i32),
            Value::I64(x) => Value::I64(x.unsigned_abs() as i64),
            Value::I128(x) => Value::I128(x.unsigned_abs() as i128),
            _ => tc_fail!(),
        },
        UnaryOperation::Double => match value {
            Value::Field(x) => Value::Field(x.double()),
            Value::Group(x) => Value::Group(x.double()),
            _ => tc_fail!(),
        },
        UnaryOperation::Inverse => match value {
            Value::Field(x) => {
                let Ok(y) = x.inverse() else {
                    halt!(span, "attempt to invert 0field");
                };
                Value::Field(y)
            }
            _ => tc_fail!(),
        },
        UnaryOperation::Negate => match value {
            Value::I8(x) => match x.checked_neg() {
                None => halt!(span, "negation overflow"),
                Some(y) => Value::I8(y),
            },
            Value::I16(x) => match x.checked_neg() {
                None => halt!(span, "negation overflow"),
                Some(y) => Value::I16(y),
            },
            Value::I32(x) => match x.checked_neg() {
                None => halt!(span, "negation overflow"),
                Some(y) => Value::I32(y),
            },
            Value::I64(x) => match x.checked_neg() {
                None => halt!(span, "negation overflow"),
                Some(y) => Value::I64(y),
            },
            Value::I128(x) => match x.checked_neg() {
                None => halt!(span, "negation overflow"),
                Some(y) => Value::I128(y),
            },
            Value::Group(x) => Value::Group(-x),
            Value::Field(x) => Value::Field(-x),
            _ => tc_fail!(),
        },
        UnaryOperation::Not => match value {
            Value::Bool(x) => Value::Bool(!x),
            Value::U8(x) => Value::U8(!x),
            Value::U16(x) => Value::U16(!x),
            Value::U32(x) => Value::U32(!x),
            Value::U64(x) => Value::U64(!x),
            Value::U128(x) => Value::U128(!x),
            Value::I8(x) => Value::I8(!x),
            Value::I16(x) => Value::I16(!x),
            Value::I32(x) => Value::I32(!x),
            Value::I64(x) => Value::I64(!x),
            Value::I128(x) => Value::I128(!x),
            _ => tc_fail!(),
        },
        UnaryOperation::Square => match value {
            Value::Field(x) => Value::Field(x.square()),
            _ => tc_fail!(),
        },
        UnaryOperation::SquareRoot => match value {
            Value::Field(x) => {
                let Ok(y) = x.square_root() else {
                    halt!(span, "square root failure");
                };
                Value::Field(y)
            }
            _ => tc_fail!(),
        },
        UnaryOperation::ToXCoordinate => match value {
            Value::Group(x) => Value::Field(x.to_x_coordinate()),
            _ => tc_fail!(),
        },
        UnaryOperation::ToYCoordinate => match value {
            Value::Group(x) => Value::Field(x.to_y_coordinate()),
            _ => tc_fail!(),
        },
    };

    Ok(value_result)
}
