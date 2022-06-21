// Copyright (C) 2019-2022 Aleo Systems Inc.
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

use std::cell::RefCell;

use leo_ast::Function;
use leo_errors::{AstError, Result};
use leo_span::{Span, Symbol};

use indexmap::IndexMap;

use crate::{FunctionSymbol, VariableSymbol};

#[derive(Clone, Debug, Default)]
pub struct SymbolTable {
    /// The parent scope if it exists.
    /// For example if we are in a if block inside a function.
    pub(crate) parent: Option<Box<SymbolTable>>,
    /// Functions represents the name of each function mapped to the Ast's function definition.
    /// This field is populated at a first pass.
    pub functions: IndexMap<Symbol, FunctionSymbol>,
    /// The variables defined in a scope.
    /// This field is populated as necessary.
    pub(crate) variables: IndexMap<Symbol, VariableSymbol>,
    scope_index: usize,
    pub(crate) scopes: Vec<RefCell<SymbolTable>>,
}

impl SymbolTable {
    pub fn check_shadowing(&self, symbol: Symbol, span: Span) -> Result<()> {
        if self.variables.contains_key(&symbol) {
            Err(AstError::shadowed_variable(symbol, span).into())
        } else if self.functions.contains_key(&symbol) {
            Err(AstError::shadowed_function(symbol, span).into())
        } else if let Some(parent) = self.parent.as_ref() {
            parent.check_shadowing(symbol, span)
        } else {
            Ok(())
        }
    }

    pub fn scope_index(&mut self) -> usize {
        let index = self.scope_index;
        self.scope_index = self.scope_index.saturating_add(1);
        index
    }

    pub fn insert_fn(&mut self, symbol: Symbol, insert: &Function) -> Result<()> {
        self.check_shadowing(symbol, insert.span)?;
        let id = self.scope_index();
        self.functions.insert(symbol, Self::new_function_symbol(id, insert));
        self.scopes.push(Default::default());
        Ok(())
    }

    pub fn insert_variable(&mut self, symbol: Symbol, insert: VariableSymbol) -> Result<()> {
        self.check_shadowing(symbol, insert.span)?;
        self.variables.insert(symbol, insert);
        Ok(())
    }

    pub fn insert_block(&mut self) -> usize {
        self.scopes.push(Default::default());
        self.scope_index()
    }

    pub fn lookup_fn(&self, symbol: Symbol) -> Option<&FunctionSymbol> {
        if let Some(func) = self.functions.get(&symbol) {
            Some(func)
        } else if let Some(parent) = self.parent.as_ref() {
            parent.lookup_fn(symbol)
        } else {
            None
        }
    }

    pub fn lookup_variable(&self, symbol: Symbol) -> Option<&VariableSymbol> {
        if let Some(var) = self.variables.get(&symbol) {
            Some(var)
        } else if let Some(parent) = self.parent.as_ref() {
            parent.lookup_variable(symbol)
        } else {
            None
        }
    }

    pub fn get_fn_scope(&self, symbol: Symbol) -> Option<&RefCell<Self>> {
        if let Some(func) = self.functions.get(&symbol) {
            self.scopes.get(func.id)
        } else if let Some(parent) = self.parent.as_ref() {
            parent.get_fn_scope(symbol)
        } else {
            None
        }
    }

    pub fn get_block_scope(&self, index: usize) -> Option<&RefCell<Self>> {
        self.scopes.get(index)
    }
}