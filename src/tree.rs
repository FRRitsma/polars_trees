use polars::prelude::Expr;


struct Tree{
    left_node: Option<Box<Tree>>,
    right_node: Option<Box<Tree>>,
    split_expression: Option<Expr>,
    is_final: bool,
    depth: u8,
    minimal_sample_size: u16,
    max_depth: u8,
}

