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

use std::path::PathBuf;

use snarkvm::prelude::{Network, ProgramID, TestnetV0};

#[cfg(not(feature = "only_testnet"))]
use snarkvm::prelude::{CanaryV0, MainnetV0};

use leo_errors::UtilError;
use leo_retriever::{Manifest, NetworkName, Retriever};
use leo_span::Symbol;

use super::*;

/// Deploys an Aleo program.
#[derive(Parser, Debug)]
pub struct LeoInterpret {
    #[clap(flatten)]
    pub(crate) compiler_options: BuildOptions,
}

impl Command for LeoInterpret {
    type Input = <LeoBuild as Command>::Output;
    type Output = ();

    fn log_span(&self) -> Span {
        tracing::span!(tracing::Level::INFO, "Leo")
    }

    fn prelude(&self, context: Context) -> Result<Self::Input> {
        (LeoBuild { options: self.compiler_options.clone() }).execute(context)
    }

    fn apply(self, context: Context, _: Self::Input) -> Result<Self::Output> {
        // Parse the network.
        let network = NetworkName::try_from(context.get_network(&self.compiler_options.network)?)?;
        match network {
            NetworkName::TestnetV0 => handle_interpret::<TestnetV0>(&self, context),
            NetworkName::MainnetV0 => {
                #[cfg(feature = "only_testnet")]
                panic!("Mainnet chosen with only_testnet feature");
                #[cfg(not(feature = "only_testnet"))]
                return handle_interpret::<MainnetV0>(&self, context);
            }
            NetworkName::CanaryV0 => {
                #[cfg(feature = "only_testnet")]
                panic!("Canary chosen with only_testnet feature");
                #[cfg(not(feature = "only_testnet"))]
                return handle_interpret::<CanaryV0>(&self, context);
            }
        }
    }
}

fn handle_interpret<N: Network>(command: &LeoInterpret, context: Context) -> Result<()> {
    // Get the package path.
    let package_path = context.dir()?;
    let home_path = context.home()?;

    // Get the program id.
    let manifest = Manifest::read_from_dir(&package_path)?;
    let program_id = ProgramID::<N>::from_str(manifest.program())?;

    // Retrieve all local dependencies in post order
    let main_sym = Symbol::intern(&program_id.name().to_string());
    let mut retriever = Retriever::<N>::new(
        main_sym,
        &package_path,
        &home_path,
        context.get_endpoint(&command.compiler_options.endpoint)?.to_string(),
    )
    .map_err(|err| UtilError::failed_to_retrieve_dependencies(err, Default::default()))?;
    let mut local_dependencies =
        retriever.retrieve().map_err(|err| UtilError::failed_to_retrieve_dependencies(err, Default::default()))?;

    // Push the main program at the end of the list.
    local_dependencies.push(main_sym);

    let paths: Vec<PathBuf> = local_dependencies
        .into_iter()
        .map(|dependency| {
            let base_path = retriever.get_context(&dependency).full_path();
            base_path.join("src/main.leo")
        })
        .collect();

    leo_interpreter::interpret(&paths)
}