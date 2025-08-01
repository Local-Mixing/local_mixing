use crate::circuit::control_functions::Gate_Control_Func;

//table consisting of all evaluations of a,b under the control functions
// control_index  a  b     index            index_formula
// -----------------------------------------------
// 0              0  0     0                (0 << 2) | (0 << 1) | 0 = 0
// 0              0  1     1                (0 << 2) | (0 << 1) | 1 = 1
// 0              1  0     2                (0 << 2) | (1 << 1) | 0 = 2
// 0              1  1     3                (0 << 2) | (1 << 1) | 1 = 3
// 1              0  0     4                (1 << 2) | (0 << 1) | 0 = 4
// 1              0  1     5                (1 << 2) | (0 << 1) | 1 = 5
// 1              1  0     6                (1 << 2) | (1 << 1) | 0 = 6
// 1              1  1     7                (1 << 2) | (1 << 1) | 1 = 7
// ...
// 15             1  1    63                (15 << 2) | (1 << 1) | 1 = 63
pub const CONTROL_FUNC_TABLE: [bool; 64] = {
    let mut table = [false; 64];
    let mut index = 0;
    while index < 64{
        //index has 6 bits. the least significant is b, next is a, then the next 4 will denote the control function
        let b = index & 1 == 1;
        let a = (index >> 1) & 1 == 1;
        let control_function = Gate_Control_Func::from_u8((index >> 2) as _);
        table[index] = control_function.evaluate(a,b);
        index +=1;
    }
    table
};