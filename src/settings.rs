#[derive(Clone, Copy)]
pub struct Settings {
    max_depth: u8,
    min_leave_size: u128,
    max_cardinality: u8,
}

impl Settings {
    pub fn new(max_depth: u8, min_leave_size: u128, max_cardinality: u8) -> Self {
        Self {
            max_depth,
            min_leave_size,
            max_cardinality,
        }
    }

    pub fn default() -> Self {
        Self::new(4, 32, 6)
    }

    pub fn set_max_depth(&mut self, max_depth: u8) {
        self.max_depth = max_depth;
    }

    pub fn get_max_depth(&self) -> u8 {
        self.max_depth
    }

    pub fn get_min_leave_size(&self) -> u128 {
        self.min_leave_size
    }

    pub fn get_max_cardinality(&self) -> u8 {
        self.max_cardinality
    }
}
