// Copyright (C) 2019-2021 Aleo Systems Inc.
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

//! The verification key file.

use crate::PackageFile;

use leo_errors::{PackageError, Result};

use serde::Deserialize;

pub static VERIFICATION_KEY_FILE_EXTENSION: &str = ".lvk";

#[derive(Deserialize)]
pub struct VerificationKeyFile {
    pub package_name: String,
}

impl VerificationKeyFile {
    pub fn new(package_name: &str) -> Self {
        Self {
            package_name: package_name.to_string(),
        }
    }

    /// Reads the verification key from the given file path if it exists.
    pub fn read_from(&self, path: &std::path::Path) -> Result<Vec<u8>> {
        let path = self.file_path(path);
        let bytes =
            std::fs::read(&path).map_err(|_| PackageError::failed_to_read_verification_key_file(path.into_owned()))?;
        Ok(bytes)
    }
}

impl PackageFile for VerificationKeyFile {
    type ParentDirectory = super::OutputsDirectory;

    fn template(&self) -> String {
        unimplemented!("PackageFile doesn't have a template.");
    }
}

impl std::fmt::Display for VerificationKeyFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.lvk", self.package_name)
    }
}