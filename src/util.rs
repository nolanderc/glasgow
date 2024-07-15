#[derive(Clone, Copy, PartialEq, Eq)]
pub struct U24([u8; 3]);

impl U24 {
    pub const ZERO: U24 = U24([0, 0, 0]);

    pub fn from_usize(x: usize) -> Option<U24> {
        if x >= (1 << 24) {
            return None;
        }
        let [a, b, c, _] = u32::to_le_bytes(x as u32);
        Some(U24([a, b, c]))
    }

    pub fn to_u32(self) -> u32 {
        let [a, b, c] = self.0;
        u32::from_le_bytes([a, b, c, 0])
    }

    pub fn to_usize(self) -> usize {
        const {
            assert!(std::mem::size_of::<u32>() <= std::mem::size_of::<usize>());
        }
        self.to_u32() as usize
    }
}

impl std::fmt::Debug for U24 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_u32().fmt(f)
    }
}

impl std::fmt::Display for U24 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_u32().fmt(f)
    }
}
