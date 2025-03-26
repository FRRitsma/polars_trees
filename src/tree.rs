use polars::prelude::Expr;


struct Tree{
    left_node: Option<Tree>,
    right_node: Option<Tree>,
    split_expression: Option<Tree>,
    is_final: bool,
    depth: u8,
    minimal_sample_size: u16,
    max_depth: u8,
}

