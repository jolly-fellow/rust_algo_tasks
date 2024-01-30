use std::cmp;
use std::cmp::max;

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
    leetcode_1137(n - 1) + leetcode_1137(n - 2) + leetcode_1137(n - 3)
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
    let n = cost.len() as isize;
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
//    C++ recursive style
pub fn leetcode_198(nums: Vec<i32>) -> i32 {
    fn r(nums: &Vec<i32>, i: isize) -> i32 {
        if i < 0 {
            0
        } else if i == 0 {
            nums[i as usize]
        } else if i == 1 {
            cmp::max(nums[i as usize], nums[i as usize - 1])
        } else {
            cmp::max(r(nums, i - 1), r(nums, i - 2) + nums[i as usize])
        }
    }
    r(&nums, nums.len() as isize - 1)
}

// Rust recursive style
pub fn leetcode_198_recursive(nums: Vec<i32>) -> i32 {
    fn r(nums: &Vec<i32>, i: isize) -> i32 {
        match i {
            i if i < 0 => 0,
            0 => nums[0],
            1 => cmp::max(nums[0], nums[1]),
            _ => cmp::max(r(nums, i - 1), r(nums, i - 2) + nums[i as usize]),
        }
    }
    r(&nums, nums.len() as isize - 1)
}

// Iterative C++ style
pub fn leetcode_198_iterative_cpp(v: &Vec<i32>) -> i32 {
    let v_size = v.len();

    if v_size == 1 {
        return v[0];
    }
    let mut prev2 = v[0];
    let mut prev1 = cmp::max(v[0], v[1]);
    let mut sum = prev1;

    for i in 2..v_size {
        sum = cmp::max(v[i] + prev2, prev1);

        prev2 = prev1;
        prev1 = sum;
    }
    sum
}

// Iterative Rust functional style
pub fn leetcode_198_iterative_rust(v: &Vec<i32>) -> i32 {
    // Base case: If there's only one house, return its value
    if v.len() == 1 { return v[0]; }
    // Initialize variables to keep track of the previous two results and the current sum
    let (prev2, prev1, _) = v.iter().skip(2).fold(
        // Initial tuple values (prev2, prev1, _)
        (v[0], cmp::max(v[0], v[1]), 0),

        // Fold function to update the tuple values in each iteration
        |(prev2, prev1, _), &current| {
            // Calculate the current sum
            let sum = cmp::max(current + prev2, prev1);
            // Update the tuple values for the next iteration
            (prev1, sum, current)
        },
    );
    // Return the maximum amount between the last two results
    cmp::max(prev1, prev2)
}

// https://leetcode.com/problems/delete-and-earn
// 740. Delete and Earn

// recursive style
pub fn leetcode_740(v: Vec<i32>) -> i32 {
    if v.len() == 1 {
        return v[0];
    }

    let me = *v.iter().max().unwrap();
    let mut n = vec![0; (me + 1) as usize];

    v.iter().for_each(|&i| n[i as usize] += i);

    fn r(nums: &Vec<i32>, i: isize) -> i32 {
        match i {
            i if i < 0 => 0,
            0 => nums[0],
            1 => max(nums[0], nums[1]),
            _ => max(r(nums, i - 1), r(nums, i - 2) + nums[i as usize]),
        }
    }
    r(&n, n.len() as isize - 1)
}


// C++ iterative style

pub fn leetcode_740_iterative_cpp(v: Vec<i32>) -> i32 {
    if v.len() == 1 {
        return v[0];
    }

    let me = *v.iter().max().unwrap();
    let mut n = vec![0; (me + 1) as usize];

    for &i in &v {
        n[i as usize] += i;
    }

    let mut prev2 = n[0];
    let mut prev1 = n[1];
    let mut sum = 0;

    for it in 2..n.len() {
        sum = max(n[it] + prev2, prev1);
        prev2 = prev1;
        prev1 = sum;
    }
    sum
}

// Functional iterative style
pub fn leetcode_740_iterative_rust(v: Vec<i32>) -> i32 {
    if v.len() == 1 {
        return v[0];
    }

    let me = *v.iter().max().unwrap();
    let mut n = vec![0; (me + 1) as usize];

    v.iter().for_each(|&i| n[i as usize] += i);

    let sum = n.iter().skip(2).fold(
        (n[0], n[1]), |(prev2, prev1), &current| {
        (prev1, max(current + prev2, prev1))
    }).1;

    sum
}

// https://leetcode.com/problems/unique-paths
// 62. Unique Paths
pub fn leetcode_62(m: i32, n: i32) -> i32 {
    let mut dp = vec![vec![0; n as usize]; m as usize];

    for r in 0..m as usize {
        for c in 0..n as usize {
            if r == 0 || c == 0 {
                dp[r][c] = 1;
            } else {
                dp[r][c] = dp[r - 1][c] + dp[r][c - 1];
            }
        }
    }
    dp[m as usize - 1][n as usize - 1]
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leetcode_62() {
        let result = leetcode_62(3,7);
        assert_eq!(result, 28);
        let result = leetcode_62(3,2);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_leetcode_740() {
        let result = leetcode_740(vec![3,4,2]);
        assert_eq!(result, 6);
        let result = leetcode_740(vec![2,2,3,3,3,4]);
        assert_eq!(result, 9);
        let result = leetcode_740_iterative_cpp(vec![3,4,2]);
        assert_eq!(result, 6);
        let result = leetcode_740_iterative_cpp(vec![2,2,3,3,3,4]);
        assert_eq!(result, 9);
        let result = leetcode_740_iterative_rust(vec![3,4,2]);
        assert_eq!(result, 6);
        let result = leetcode_740_iterative_rust(vec![2,2,3,3,3,4]);
        assert_eq!(result, 9);

    }

    #[test]
    fn test_leetcode_198() {
        let long_vec = vec![114, 117, 207, 117, 235, 82, 90, 67, 143, 146, 53, 108,
                            200, 91, 80, 223, 58, 170, 110, 236, 81, 90, 222, 160, 165, 195, 187,
                            199, 114, 235, 197, 187, 69, 129, 64, 214, 228, 78, 188, 67, 205, 94,
                            205, 169, 241, 202, 144, 240];
        let long_vec_result = 4173;
        let result = leetcode_198(vec![1, 2, 3, 1]);
        assert_eq!(result, 4);
        let result = leetcode_198(vec![2, 7, 9, 3, 1]);
        assert_eq!(result, 12);
        let result = leetcode_198_recursive(vec![1, 2, 3, 1]);
        assert_eq!(result, 4);
        let result = leetcode_198_recursive(vec![2, 7, 9, 3, 1]);
        assert_eq!(result, 12);
        let result = leetcode_198_iterative_cpp(&vec![1, 2, 3, 1]);
        assert_eq!(result, 4);
        let result = leetcode_198_iterative_cpp(&vec![2, 7, 9, 3, 1]);
        assert_eq!(result, 12);
        let result = leetcode_198_iterative_cpp(&long_vec);
        assert_eq!(result, long_vec_result);
        let result = leetcode_198_iterative_rust(&vec![1, 2, 3, 1]);
        assert_eq!(result, 4);
        let result = leetcode_198_iterative_rust(&vec![2, 7, 9, 3, 1]);
        assert_eq!(result, 12);
        let result = leetcode_198_iterative_rust(&long_vec);
        assert_eq!(result, long_vec_result);
    }

    #[test]
    fn test_leetcode_746() {
        let result = leetcode_746_iterative(vec![10, 15, 20]);
        assert_eq!(result, 15);
        let result = leetcode_746(vec![10, 15, 20]);
        assert_eq!(result, 15);
        let result = leetcode_746(vec![1, 100, 1, 1, 1, 100, 1, 1, 100, 1]);
        assert_eq!(result, 6);
        let result = leetcode_746_iterative(vec![1, 100, 1, 1, 1, 100, 1, 1, 100, 1]);
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
