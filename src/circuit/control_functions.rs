use std::error::Error;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gate_Control_Func {
    F = 0,     // false,
    AND = 1,   // a & b,
    ANDNB = 2, // a & (!b), this is r57
    A = 3,     // a,
    ANDNA = 4, // (!a) & b,
    B = 5,     // b,
    XOR = 6,   // a ^ b,
    OR = 7,    // a | b,
    NOR = 8,   // !(a | b),
    EQUIV = 9, // (a & b) | ((!a) & (!b)),
    NB = 10,   // !b,
    ORNB = 11, // (!b) | a,
    NA = 12,   // !a,
    ORNA = 13, // (!a) | b,
    NAND = 14, // !(a & b),
    T = 15,    // true,
}

impl Gate_Control_Func {
    pub const fn from_u8(u: u8) -> Self {
        match u {
            0 => Self::F,
            1 => Self::AND,
            2 => Self::ANDNB,
            3 => Self::A,
            4 => Self::ANDNA,
            5 => Self::B,
            6 => Self::XOR,
            7 => Self::OR,
            8 => Self::NOR,
            9 => Self::EQUIV,
            10 => Self::NB,
            11 => Self::ORNB,
            12 => Self::NA,
            13 => Self::ORNA,
            14 => Self::NAND,
            15 => Self::T,
            _ => unreachable!(),
        }
    }

    pub const fn evaluate(&self, a: bool, b: bool) -> bool {
        match self {
            Self::F => false,
            Self::AND => a & b,
            Self::ANDNB => a & (!b),
            Self::A => a,
            Self::ANDNA => (!a) & b,
            Self::B => b,
            Self::XOR => a ^ b,
            Self::OR => a | b,
            Self::NOR => !(a | b),
            Self::EQUIV => (a & b) | ((!a) & (!b)),
            Self::NB => !b,
            Self::ORNB => (!b) | a,
            Self::NA => !a,
            Self::ORNA => (!a) | b,
            Self::NAND => !(a & b),
            Self::T => true,
        }
    }

    pub const fn not(u: u8) -> u8 {
         match u {
            0 => 15, // F -> T
            1 => 14, // AND -> NAND
            2 => 13, // ANDNB -> ORNA
            3 => 12, // A -> NA
            4 => 11, // ANDNA -> ORNB
            5 => 10, // B -> NB
            6 => 9,  // XOR -> EQUIV
            7 => 8,  // OR -> NOR
            8 => 7,  // NOR -> OR
            9 => 6,  // EQUIV -> XOR
            10 => 5, // NB -> B
            11 => 4, // ORNB -> ANDNA
            12 => 3, // NA -> A
            13 => 2, // ORNA -> ANDNB
            14 => 1, // NAND -> AND
            15 => 0, // T -> F
            _ => unreachable!(),
        }
    }
    pub fn to_string(&self) -> String {
        match self {
            Self::F => "0".to_string(),
            Self::AND => "a&b".to_string(),
            Self::ANDNB => "a&!b".to_string(),
            Self::A => "a".to_string(),
            Self::ANDNA => "!a&b".to_string(),
            Self::B => "b".to_string(),
            Self::XOR => "a^b".to_string(),
            Self::OR => "a|b".to_string(),
            Self::NOR => "!(a|b)".to_string(),
            Self::EQUIV => "a=b".to_string(),
            Self::NB => "!b".to_string(),
            Self::ORNB => "!b|a".to_string(),
            Self::NA => "!a".to_string(),
            Self::ORNA => "!a|b".to_string(),
            Self::NAND => "!(a&b)".to_string(),
            Self::T => "1".to_string(),
        }
    }
}