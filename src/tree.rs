use polars::prelude::Expr;


struct Tree{
    left_node: Some(Tree),
    right_node: Some(Tree),
    split_expression: Some(Expr),
    is_final: bool,
    depth: u8,
    minimal_sample_size: u16,
}

