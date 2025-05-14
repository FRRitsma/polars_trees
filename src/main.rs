use std::error::Error;

mod classification_tree;
mod constants;
mod display_tree;
mod empty_tree;
mod filler_strings;
mod gini_impurity;
mod old_preprocessing;
mod settings;
mod test_utils;
mod lib;

fn main() -> Result<(), Box<dyn Error>> {
    Ok(())
}
