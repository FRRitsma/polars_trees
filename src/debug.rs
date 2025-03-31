use crate::generic_functions::get_leaf_value_for_column;
use crate::test_utils::get_preprocessed_test_dataframe;

#[test]
fn debug(){
    let df = get_preprocessed_test_dataframe();
    let schema = df.logical_plan.compute_schema().unwrap();
    let feature_column = "Ticket";
    let leaf_value = get_leaf_value_for_column(&df, &schema, feature_column).unwrap();
    println!("{}", leaf_value.information_value);
}
