/*
https://leetcode.com/problems/calculate-money-in-leetcode-bank/description/
 */
pub fn leetcode_1716(n: i32) -> i32 {
    let k = n / 7; // number of weeks
    let m = n % 7; // rest of the last week
    7 * ((k * (k + 1) / 2) + 3 * k) + (m * (m + 1) / 2) + m * k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leetcode_1716() {
        let result = leetcode_1716(4);
        assert_eq!(result, 10);
        let result = leetcode_1716(10);
        assert_eq!(result, 37);
        let result = leetcode_1716(20);
        assert_eq!(result, 96);
    }
}
