use crate::constants::TARGET_COLUMN;
use crate::settings::Settings;

use polars_core::prelude::NamedFrom;

use crate::filler_strings::rename_filler_string_full_lazyframe;
use polars_lazy::frame::LazyFrame;
use std::error::Error;

pub const REDUNDANT_STRING_VALUE: &str = "FILLER_STRING";

pub fn pre_process_dataframe(lf: LazyFrame, settings: Settings, target_column: &str) -> LazyFrame {
    let mut lf = lf.rename([target_column], [TARGET_COLUMN], true);
    lf = rename_filler_string_full_lazyframe(lf, settings).unwrap();
    lf
}

#[cfg(test)]
mod tests {
    use crate::filler_strings::rename_filler_string_full_lazyframe;
    use crate::settings::Settings;
    use crate::test_utils::get_raw_test_dataframe;

    #[test]
    fn test_rename_filler_strings() -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            std::env::set_var("POLARS_FMT_MAX_COLS", "100");
            std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
        }
        let lf = get_raw_test_dataframe();

        let renamed_lf = rename_filler_string_full_lazyframe(lf, Settings::default())?;

        let collected_lf = renamed_lf.collect()?;
        let unique_sex = collected_lf.column("Sex")?.n_unique()?;
        assert_eq!(unique_sex, 2);

        Ok(())
    }
}
