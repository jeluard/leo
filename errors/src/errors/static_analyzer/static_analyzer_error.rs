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

use crate::create_messages;
use std::fmt::{Debug, Display};

// TODO: Consolidate errors.

create_messages!(
    /// StaticAnalyzer enum that represents all the errors for static analysis.
    StaticAnalyzerError,
    code_mask: 4000i32,
    code_prefix: "SAZ",

    @formatted
    no_path_awaits_all_futures_exactly_once {
        args: (num_total_paths: impl Display),
        msg: format!("Futures must be awaited exactly once. Out of `{num_total_paths}`, there does not exist a single path in which all futures are awaited exactly once."),
        help: Some("Ex: for `f: Future` call `f.await()` to await a future. Remove duplicate future await redundancies, and add future awaits for un-awaited futures.".to_string()),
    }

    @formatted
    future_awaits_missing {
        args: (unawaited: impl Display),
        msg: format!("The following futures were never awaited: {unawaited}"),
        help: Some("Ex: for `f: Future` call `f.await()` to await a future.".to_string()),
    }

    @formatted
    invalid_await_call {
        args: (),
        msg: "Not a valid await call.".to_string(),
        help: Some("Ex: for `f: Future` call `f.await()` or `Future::await(f)` to await a future.".to_string()),
    }

    @formatted
    expected_future {
        args: (type_: impl Display),
        msg: format!("Expected a future, but found `{type_}`"),
        help: Some("Only futures can be awaited.".to_string()),
    }

    @formatted
    async_transition_call_with_future_argument {
        args: (function_name: impl Display),
        msg: format!("The call to {function_name} will result in failed executions on-chain."),
        help: Some("There is a subtle error that occurs if an async transition call follows a non-async transition call, and the async call returns a `Future` that itself takes a `Future` as an input. See See `https://github.com/AleoNet/snarkVM/issues/2570` for more context.".to_string()),
    }
);