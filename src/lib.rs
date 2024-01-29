use std::cmp;

/*
https://leetcode.com/problems/calculate-money-in-leetcode-bank/description/
*/

pub fn leetcode_1716(n: i32) -> i32 {
    let k = n / 7; // number of weeks
    let m = n % 7; // rest of the last week
    let w = 7 * ((k * (k + 1) / 2) + 3 * k); // sum for full weeks
    let p = (m * (m + 1) / 2) + m * k; // sum for partial week
    w + p
}

/*
 https://leetcode.com/problems/climbing-stairs

Actually we need to calcolate the Fibonacci sequence because number of
possible ways to up to the stairs is sum of N+1 stairs and N+2 stairs where N is a number of stairs

n can't be more than 45 amd less than 0

*/

//     Recursive solution
pub fn leetcode_70(n: i32) -> i32 {
    if n < 4 {
        return n;
    }
    leetcode_70(n - 1) + leetcode_70(n - 2)
}

//     Iterative solution with array
pub fn leetcode_70_array(n: usize) -> usize {
    if n < 4 {
        n
    } else {
        let mut v: [usize; 46] = [0; 46];
        v[0] = 1;
        v[1] = 2;
        for i in 2..=n {
            v[i] = v[i - 1] + v[i - 2];
        }
        v[n - 1]
    }
}

//     Iterative solution full optimized
pub fn leetcode_70_full(n: usize) -> usize {
    if n < 4 {
        return n;
    }

    let mut prev1 = 1;
    let mut prev2 = 2;
    let mut ways = 0;

    for _ in 2..n {
        ways = prev1 + prev2;
        prev1 = prev2;
        prev2 = ways;
    }
    ways
}

// https://leetcode.com/problems/fibonacci-number/
// 509. Fibonacci Number

pub fn leetcode_509(n: i32) -> i32 {
    if n < 2 {
        return n;
    }
    let mut prev1 = 0;
    let mut prev2 = 1;
    let mut res = prev1 + prev2;

    for _ in 2..=n {
        res = prev1 + prev2;
        prev1 = prev2;
        prev2 = res;
    }
    res
}

// https://leetcode.com/problems/n-th-tribonacci-number
// 1137. N-th Tribonacci Number recursive solution
pub fn leetcode_1137(n: i32) -> i32 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    if n == 2 { return 1; }
    leetcode_1137(n-1) + leetcode_1137(n-2) + leetcode_1137(n-3)
}
// 1137. N-th Tribonacci Number iterative solution
pub fn leetcode_1137_iterative(n: i32) -> i32 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    if n == 2 { return 1; }
    let mut sum = 0;
    let mut prev1 = 0;
    let mut prev2 = 1;
    let mut prev3 = 1;

    for _ in 2..n {
        sum = prev1 + prev2 + prev3;
        prev1 = prev2;
        prev2 = prev3;
        prev3 = sum;
    }
    sum
}

// https://leetcode.com/problems/min-cost-climbing-stairs/
// 746. Min Cost Climbing Stairs

// recursive
pub fn leetcode_746(cost: Vec<i32>) -> i32 {
    fn r(cost: &Vec<i32>, i: isize) -> i32 {
        match i {
            i if i < 0 => 0,
            i if i < 2 => cost[i as usize],
            _ => cost[i as usize] + cmp::min(r(cost, i - 1), r(cost, i - 2)),
        }
    }
    let n = cost.len()  as isize;
    cmp::min(r(&cost, n - 1), r(&cost, n - 2))
}

pub fn leetcode_746_iterative(cost: Vec<i32>) -> i32 {
    let mut prev = 0;
    let mut prevprev = 0;
    let mut current = 0;

    for i in 2..=cost.len() {
        current = cmp::min(cost[i - 1] + prev, cost[i - 2] + prevprev);
        prevprev = prev;
        prev = current;
    }
    current
}

// https://leetcode.com/problems/house-robber/
// 198. House Robber


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leetcode_746() {
        let result = leetcode_746_iterative(vec![10,15,20]);
        assert_eq!(result, 15);
        let result = leetcode_746(vec![10,15,20]);
        assert_eq!(result, 15);
        let result = leetcode_746(vec![1,100,1,1,1,100,1,1,100,1]);
        assert_eq!(result, 6);
        let result = leetcode_746_iterative(vec![1,100,1,1,1,100,1,1,100,1]);
        assert_eq!(result, 6);
    }
    #[test]
    fn test_leetcode_1716() {
        let result = leetcode_1716(4);
        assert_eq!(result, 10);
        let result = leetcode_1716(10);
        assert_eq!(result, 37);
        let result = leetcode_1716(20);
        assert_eq!(result, 96);
    }
    #[test]
    fn test_leetcode_70() {
        assert_eq!(leetcode_70(20), 10946);
    }
    #[test]
    fn test_leetcode_70_array() {
        assert_eq!(leetcode_70_array(20), 10946);
    }
    #[test]
    fn test_leetcode_70_full() {
        assert_eq!(leetcode_70_full(20), 10946);
    }

    #[test]
    fn test_leetcode_509() {
        let result = leetcode_509(2);
        assert_eq!(result, 1);
        let result = leetcode_509(3);
        assert_eq!(result, 2);
        let result = leetcode_509(4);
        assert_eq!(result, 3);
        let result = leetcode_509(10);
        assert_eq!(result, 55);
    }
    #[test]
    fn test_leetcode_1137() {
        let result = leetcode_1137(3);
        assert_eq!(result, 2);
        let result = leetcode_1137(4);
        assert_eq!(result, 4);
        let result = leetcode_1137(5);
        assert_eq!(result, 7);
    }
    #[test]
    fn test_leetcode_1137_iterative() {
        let result = leetcode_1137_iterative(3);
        assert_eq!(result, 2);
        let result = leetcode_1137_iterative(4);
        assert_eq!(result, 4);
        let result = leetcode_1137_iterative(5);
        assert_eq!(result, 7);
        let result = leetcode_1137_iterative(25);
        assert_eq!(result, 1389537);

    }
}
