use std::cmp;
use std::cmp::max;
use std::cmp::min;
use std::collections::HashSet;


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

    for i in v.iter().skip(2) {
        sum = cmp::max(i + prev2, prev1);

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

    for &element in n.iter().skip(2) {
        sum = max(element + prev2, prev1);
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

pub fn leetcode_62_recursive(m: i32, n: i32) -> i32 {
    if m == 1 || n == 1 {
        1
    } else {
        leetcode_62_recursive(m - 1, n) + leetcode_62_recursive(m, n - 1)
    }
}



// https://leetcode.com/problems/minimum-path-sum
// 64. Minimum Path Sum
pub fn leetcode_64(grid: Vec<Vec<i32>>) -> i32 {
    fn r(g: &Vec<Vec<i32>>, row: usize, col: usize) -> i32 {
        if row == 0 && col == 0 {
            g[0][0]
        } else if row == 0 {
            g[row][col] + r(g, row, col - 1)
        } else if col == 0 {
            g[row][col] + r(g, row - 1, col)
        } else {
            g[row][col] + min(
                r(g, row - 1, col),
                r(g, row, col - 1),
            )
        }
    }
    r(&grid, grid.len()-1, grid[0].len()-1)
}

pub fn leetcode_64_iterative(grid: Vec<Vec<i32>>) -> i32 {
    let rows = grid.len();
    let cols = grid[0].len();

    let mut dp = vec![i32::MAX; cols + 1];
    dp[1] = 0;

    for row in 1..=rows {
        for col in 1..=cols {
            dp[col] = grid[row - 1][col - 1] + dp[col].min(dp[col - 1]);
        }
    }
    dp[cols]
}

// https://leetcode.com/problems/triangle/
// 120. Triangle

pub fn leetcode_120(triangle: Vec<Vec<i32>>) -> i32 {
    let n = triangle.len();
    let mut minlen = triangle.last().unwrap().clone();

    for layer in (0..n - 1).rev() {
        for i in 0..=layer {
            // Find the lesser of its two children, and sum the current value in the triangle with it.
            minlen[i] = minlen[i].min(minlen[i + 1]) + triangle[layer][i];
        }
    }
    minlen[0]
}

// https://leetcode.com/problems/unique-paths-ii/description/
// 63. Unique Paths II
pub fn leetcode_63(grid: Vec<Vec<i32>>) -> i32 {
    fn r(m: i32, n: i32, g: & Vec<Vec<i32>>) -> i32 {
        if m > g.len() as i32-1 || n > g[0].len() as i32-1 || m < 0 || n < 0 || g[m as usize][n as usize] == 1 {
            0
        }
        else if m == g.len() as i32-1 && n == g[0].len() as i32-1 {
            1
        }
        else {
            r(m+1, n, g) + r(m, n+1, g)
        }
    }
    r(0, 0, &grid)
}

pub fn leetcode_63_memo(grid: &Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![0; grid[0].len()]; grid.len()];

    fn r(g: &Vec<Vec<i32>>, dp: &mut Vec<Vec<i32>>, i: usize, j: usize) -> i32 {
        let m = g.len();
        let n = g[0].len();

        if i >= m || j >= n || i == usize::MAX || j == usize::MAX || g[i][j] == 1 {
            return 0;
        }
        if i == m - 1 && j == n - 1 {
            return 1;
        }
        if dp[i][j] != 0 {
            return dp[i][j];
        }
        dp[i][j] = r(g, dp, i + 1, j) + r(g, dp, i, j + 1);
        dp[i][j]
    }
    r(grid, &mut dp, 0, 0)
}

pub fn leetcode_63_iterative(grid: &Vec<Vec<i32>>) -> i32 {
    let m = grid.len() + 1;
    let n = grid[0].len() + 1;

    let mut dp: Vec<Vec<i32>> = vec![vec![0; n]; m];

    dp[0][1] = 1;

    for i in 1..m {
        for j in 1..n {
            if grid[i - 1][j - 1] == 1 {
                dp[i][j] = 0;
            } else {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }

    dp[m - 1][n - 1]
}

// https://leetcode.com/problems/minimum-falling-path-sum/
// 931. Minimum Falling Path Sum
/*
// less optimized but looking nice approach
pub fn leetcode_931(mut grid: Vec<Vec<i32>>) -> i32 {
    for i in 1..grid.len() {
        for j in 0..grid[i].len() {
            // Creates a list of valid neighbors for the current cell (i, j) i.e.
            // 3 cells under cell (i, j),
            // ensuring they stay within grid boundaries.
            let neighbors = [
                (i - 1, j),
                (i - 1, j.saturating_sub(1)),
                (i - 1, (j + 1).min(grid[i].len() - 1)),
            ];
            grid[i][j] += neighbors.iter().map(|&(r, c)| grid[r][c]).min().unwrap();
        }
    }
    *grid.last().unwrap().iter().min().unwrap()
}*/

pub fn leetcode_931(mut grid: Vec<Vec<i32>>) -> i32 {
    let rows = grid.len();
    let cols = grid[0].len();

    for i in 1..rows {
        for j in 0..cols {
            grid[i][j] += min(grid[i - 1][j],
                              min(grid[i - 1][j.saturating_sub(1)],
                                  grid[i - 1][min(cols - 1, j + 1)]));
        }
    }
    *grid.last().unwrap().iter().min().unwrap()
}

// https://leetcode.com/problems/maximal-square/
// 221. Maximal Square
pub fn leetcode_221(matrix: & Vec<Vec<char>>) -> i32 {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut dp = vec![vec![0; cols + 1]; rows + 1];
    let mut max_side = 0;

    for r in 0..rows {
        for c in 0..cols {
            if matrix[r][c] == '1' {
                dp[r+1][c+1] = dp[r][c].min(dp[r+1][c]).min(dp[r][c+1]) + 1;
                max_side = max_side.max(dp[r+1][c+1]);
            }
        }
    }
    max_side * max_side
}


// optimized solution by memory size using a vector for dp instead of matrix because
// we don't need to keep processed lines of the given matrix.

pub fn leetcode_221_vector(matrix: & Vec<Vec<char>>) -> i32 {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut dp = vec![0; cols + 1];
    let mut prev = 0;
    let mut max_side = 0;

    for r in 0..rows {
        for c in 0..cols {
            let temp = dp[c + 1];
            if matrix[r][c] == '1' {
                dp[c + 1] = dp[c].min(dp[c + 1]).min(prev) + 1;
                max_side = max_side.max(dp[c + 1]);
            } else {
                dp[c + 1] = 0;
            }
            prev = temp;
        }
    }
    max_side * max_side
}

// Another approach with iterators
pub fn leetcode_221_vector_iter(matrix: & Vec<Vec<char>>) -> i32 {
    let mut dp = vec![0; matrix[0].len() + 1];
    let mut max_side = 0;

    for row in matrix {
        let mut prev = 0;
        // Iterate over each cell in the row along with its index
        for (col, &cell) in row.iter().enumerate() {
            let temp = dp[col + 1];
            if cell == '1' {
                dp[col + 1] = dp[col].min(dp[col + 1]).min(prev) + 1;
                max_side = max_side.max(dp[col + 1]);
            } else {
                dp[col + 1] = 0;
            }
            prev = temp;
        }
    }
    max_side * max_side
}

// https://leetcode.com/problems/longest-palindromic-substring/description/
// 5. Longest Palindromic Substring

// the simplest recursive solution
pub fn leetcode_5(s: String) -> String {
    fn longest_palindrome_substring(s: &str, i: usize, j: usize) -> &str {
        fn is_palindrome(s: &str, i: usize, j: usize) -> bool {
            if i > j {
                return true; // Empty string is considered a palindrome
            }
            if s.chars().nth(i) != s.chars().nth(j) {
                return false;
            }
            is_palindrome(s, i + 1, j - 1)
        }

        if i > j {
            return "";
        }
        if i == j {
            return &s[i..=i];
        }
        // Check if current substring is a palindrome
        if is_palindrome(s, i, j) {
            return &s[i..=j];
        }
        // Explore both possibilities: excluding first or last character
        let left = longest_palindrome_substring(s, i + 1, j);
        let right = longest_palindrome_substring(s, i, j - 1);

        // Return the longest substring found
        if left.len() >= right.len() {
            left
        } else {
            right
        }
    }
    longest_palindrome_substring(s.as_str(), 0, s.len() - 1).to_string()
}

// Manacher algorithm O(n)
// https://en.wikipedia.org/wiki/Longest_palindromic_substring

/*
string longestPalindrome(const string &s){
    vector<char> s2(s.size() * 2 + 1, '#');
    //создаем псевдостроку с границами в виде символа '#'
    for(int i = 0; i != s.size(); ++i){
        s2[i*2 + 1] = s[i];
    }

    int p[s2.size()];
    int r, c, index, i_mir;
    int maxLen = 1;
    i_mir = index = r = c = 0;

    for(int i = 1; i != s2.size() - 1; ++i){
        i_mir = 2*c-i;

        //Если мы попадаем в пределы радиуса прошлого результата,
        //то начальное значение текущего радиуса можно взять из зеркального результата
        p[i] = r > i ? min(p[i_mir], r-i) : 0;

        while(i > p[i] && (i + p[i] + 1) < s2.size() && s2[i - p[i] - 1] == s2[i + p[i] + 1])
            ++p[i];

        if(p[i] + i > r){
            c = i;
            r = i + p[i];
        }
        if(maxLen < p[i]){
            maxLen = p[i];
            index = i;
        }
    }
    //Отображаем найденные позиции на оригинальную строку
    return s.substr((index-maxLen)/2, maxLen);
}
 */
pub fn leetcode_5_manacher(s: String) -> String {
    if s.len() <= 1 { return s; }

    // MEMO: We need to detect odd palindrome as well,
    // therefore, inserting dummy string so that
    // we can find a pair with dummy center character.
    let mut chars: Vec<char> = Vec::with_capacity(s.len() * 2 + 1);
    for c in s.chars() {
        chars.push('#');
        chars.push(c);
    }
    chars.push('#');

    // List: storing the length of palindrome at each index of string
    let mut length_of_palindrome = vec![1usize; chars.len()];
    // Integer: Current checking palindrome's center index
    let mut current_center: usize = 0;
    // Integer: Right edge index existing the radius away from current center
    let mut right_from_current_center: usize = 0;

    for i in 0..chars.len() {
        // 1: Check if we are looking at right side of palindrome.
        if right_from_current_center > i && i > current_center {
            // 1-1: If so copy from the left side of palindrome.
            // If the value + index exceeds the right edge index, we should cut and check palindrome later #3.
            length_of_palindrome[i] = std::cmp::min(
                right_from_current_center - i,
                length_of_palindrome[2 * current_center - i],
            );
            // 1-2: Move the checking palindrome to new index if it exceeds the right edge.
            if length_of_palindrome[i] + i >= right_from_current_center {
                current_center = i;
                right_from_current_center = length_of_palindrome[i] + i;
                // 1-3: If radius exceeds the end of list, it means checking is over.
                // You will never get the larger value because the string will get only shorter.
                if right_from_current_center >= chars.len() - 1 {
                    break;
                }
            } else {
                // 1-4: If the checking index doesn't exceeds the right edge,
                // it means the length is just as same as the left side.
                // You don't need to check anymore.
                continue;
            }
        }

        // Integer: Current radius from checking index
        // If it's copied from left side and more than 1,
        // it means it's ensured so you don't need to check inside radius.
        let mut radius: usize = (length_of_palindrome[i] - 1) / 2;
        radius += 1;
        // 2: Checking palindrome.
        // Need to care about overflow usize.
        while i >= radius && i + radius <= chars.len() - 1 && chars[i - radius] == chars[i + radius]
        {
            length_of_palindrome[i] += 2;
            radius += 1;
        }
    }

    // 3: Find the maximum length and generate answer.
    let center_of_max = length_of_palindrome
        .iter()
        .enumerate()
        .max_by_key(|(_, &value)| value)
        .map(|(idx, _)| idx)
        .unwrap();
    let radius_of_max = (length_of_palindrome[center_of_max] - 1) / 2;
    let answer = &chars[(center_of_max - radius_of_max)..(center_of_max + radius_of_max + 1)]
        .iter()
        .collect::<String>();
    answer.replace('#', "")
}

// https://leetcode.com/problems/word-break
// 139. Word Break

pub fn leetcode_139(s: String, word_dict: Vec<String>) -> bool {

    // Convert dictionary words to a HashSet for efficient lookups
    let word_set: HashSet<String> = word_dict.into_iter().collect();

    // Find the longest word in the dictionary for boundary checks
    let longest_word = word_set.iter().map(|word| word.len()).max().unwrap_or(0);

    // Create a boolean array to track if substrings can be formed
    let mut dp: Vec<bool> = vec![false; s.len() + 1];
    // Base case: empty string can be formed
    dp[0] = true;

    // Iterate over all possible substring lengths (1 to max)
    for i in 1..=s.len() {
        // Iterate over potential ending indices of substrings in reverse order
        // This avoids redundant checks and leverages previous results
        for j in (0..i).rev().take(longest_word) {
            // Check if substring can be formed from previous valid substring and a word in the dictionary
            if dp[j] && word_set.contains(&s[j..i]) {
                // Substring can be formed, mark it as valid and break to next length
                dp[i] = true;
                break;
            }
        }
    }
    // Final element of dp tells if entire string can be formed
    dp[s.len()]
}

// Recursive solution
pub fn leetcode_139_recursive(s: String, word_dict: Vec<String>) -> bool {
    fn word_break_recursive(s: &str, dict: &HashSet<String>, idx: usize) -> bool {
        if idx == s.len() {
            return true; // Reached the end, valid word break
        }
        for word in dict {
            if s.get(idx..idx + word.len()) == Some(word) {
                // Check if remaining substring can be broken as well
                if word_break_recursive(s, dict, idx + word.len()) {
                    return true;
                }
            }
        }
        false // No valid word break found starting at `idx`
    }

    let word_set: HashSet<String> = word_dict.into_iter().collect();

    word_break_recursive(&s, &word_set, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! vec_of_strings {
        ($($x:expr),*) => (vec![$($x.to_string()),*]);
    }

    #[test]
    fn test_leetcode_139() {
        let result = leetcode_139("leetcode".to_string(), vec_of_strings!["leet", "code"]);
        assert!(result);
        let result = leetcode_139("applepenapple".to_string(), vec_of_strings!["apple", "pen"]);
        assert!(result);
        let result = leetcode_139("catsandog".to_string(), vec_of_strings!["cats","dog","sand","and","cat"]);
        assert!(!result);
        let result = leetcode_139_recursive("leetcode".to_string(), vec_of_strings!["leet", "code"]);
        assert!(result);
        let result = leetcode_139_recursive("applepenapple".to_string(), vec_of_strings!["apple", "pen"]);
        assert!(result);
        let result = leetcode_139_recursive("catsandog".to_string(), vec_of_strings!["cats","dog","sand","and","cat"]);
        assert!(!result);
    }

    #[test]
    fn test_leetcode_5() {
        let result = leetcode_5("babad".to_string());
        assert!(result == "bab" || result == "aba");
        let result = leetcode_5("cbbd".to_string());
        assert_eq!(result, "bb");
        let result = leetcode_5("a".to_string());
        assert_eq!(result, "a");

        let result = leetcode_5_manacher("babad".to_string());
        assert!(result == "bab" || result == "aba");
        let result = leetcode_5_manacher("cbbd".to_string());
        assert_eq!(result, "bb");
        let result = leetcode_5_manacher("a".to_string());
        assert_eq!(result, "a");
    }

    #[test]
    fn test_leetcode_221() {
        let grid = vec![vec!['1','0','1','0','0'],
                          vec!['1','0','1','1','1'],
                          vec!['1','1','1','1','1'],
                          vec!['1','0','0','1','0']];

        let result = leetcode_221(&grid);
        assert_eq!(result, 4);
        let result = leetcode_221(&vec![vec!['0','1'], vec!['1','0']]);
        assert_eq!(result, 1);
        let result = leetcode_221(&vec![vec!['0']]);
        assert_eq!(result, 0);
        let result = leetcode_221_vector(&grid);
        assert_eq!(result, 4);
        let result = leetcode_221_vector(&vec![vec!['0','1'], vec!['1','0']]);
        assert_eq!(result, 1);
        let result = leetcode_221_vector(&vec![vec!['0']]);
        assert_eq!(result, 0);
        let result = leetcode_221_vector_iter(&grid);
        assert_eq!(result, 4);
        let result = leetcode_221_vector_iter(&vec![vec!['0','1'], vec!['1','0']]);
        assert_eq!(result, 1);
        let result = leetcode_221_vector_iter(&vec![vec!['0']]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_leetcode_931() {
        let result = leetcode_931(vec![vec![2,1,3],vec![6,5,4],vec![7,8,9]]);
        assert_eq!(result, 13);
        let result = leetcode_931(vec![vec![-19,57],vec![-40,-5]]);
        assert_eq!(result, -59);
    }

    #[test]
    fn test_leetcode_63() {
        let big_grid =  vec![vec![0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0],
                             vec![0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                             vec![1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,0,0],
                             vec![1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1],
                             vec![0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                             vec![0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0],
                             vec![0,0,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
                             vec![1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,1],
                             vec![0,0,0,0,1,0,0,1,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                             vec![0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                             vec![1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,0],
                             vec![0,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1],
                             vec![0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0],
                             vec![1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             vec![0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             vec![0,1,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
                             vec![0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
                             vec![0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             vec![0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             vec![0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                             vec![0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0],
                             vec![0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                             vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                             vec![0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                             vec![1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             vec![0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0],
                             vec![0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0],
                             vec![0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]];
        let big_grid_result = 718991952;
        let result = leetcode_63(vec![vec![0,0,0],vec![0,1,0],vec![0,0,0]]);
        assert_eq!(result, 2);
        let result = leetcode_63(vec![vec![0,1],vec![0,0]]);
        assert_eq!(result, 1);
        let result = leetcode_63(vec![vec![0,0],vec![0,1]]);
        assert_eq!(result, 0);
        let result = leetcode_63_memo(&vec![vec![0,0,0],vec![0,1,0],vec![0,0,0]]);
        assert_eq!(result, 2);
        let result = leetcode_63_memo(&vec![vec![0,1],vec![0,0]]);
        assert_eq!(result, 1);
        let result = leetcode_63_memo(&vec![vec![0,0],vec![0,1]]);
        assert_eq!(result, 0);
        let result = leetcode_63_memo(&big_grid);
        assert_eq!(result, big_grid_result);
        let result = leetcode_63_iterative(&vec![vec![0,0,0],vec![0,1,0],vec![0,0,0]]);
        assert_eq!(result, 2);
        let result = leetcode_63_iterative(&vec![vec![0,1],vec![0,0]]);
        assert_eq!(result, 1);
        let result = leetcode_63_iterative(&vec![vec![0,0],vec![0,1]]);
        assert_eq!(result, 0);
        let result = leetcode_63_iterative(&big_grid);
        assert_eq!(result, big_grid_result);

    }
    #[test]
    fn test_leetcode_120() {
        let result = leetcode_120(vec![vec![2],vec![3,4],vec![6,5,7],vec![4,1,8,3]]);
        assert_eq!(result, 11);
        let result = leetcode_120(vec![vec![-10]]);
        assert_eq!(result, -10);
    }
    #[test]
    fn test_leetcode_64() {
        let big_grid =  vec![vec![3,8,6,0,5,9,9,6,3,4,0,5,7,3,9,3],
                             vec![0,9,2,5,5,4,9,1,4,6,9,5,6,7,3,2],
                             vec![8,2,2,3,3,3,1,6,9,1,1,6,6,2,1,9],
                             vec![1,3,6,9,9,5,0,3,4,9,1,0,9,6,2,7],
                             vec![8,6,2,2,1,3,0,0,7,2,7,5,4,8,4,8],
                             vec![4,1,9,5,8,9,9,2,0,2,5,1,8,7,0,9],
                             vec![6,2,1,7,8,1,8,5,5,7,0,2,5,7,2,1],
                             vec![8,1,7,6,2,8,1,2,2,6,4,0,5,4,1,3],
                             vec![9,2,1,7,6,1,4,3,8,6,5,5,3,9,7,3],
                             vec![0,6,0,2,4,3,7,6,1,3,8,6,9,0,0,8],
                             vec![4,3,7,2,4,3,6,4,0,3,9,5,3,6,9,3],
                             vec![2,1,8,8,4,5,6,5,8,7,3,7,7,5,8,3],
                             vec![0,7,6,6,1,2,0,3,5,0,8,0,8,7,4,3],
                             vec![0,4,3,4,9,0,1,9,7,7,8,6,4,6,9,5],
                             vec![6,5,1,9,9,2,2,7,4,2,7,2,2,3,7,2],
                             vec![7,1,9,6,1,2,7,0,9,6,6,4,4,5,1,0],
                             vec![3,4,9,2,8,3,1,2,6,9,7,0,2,4,2,0],
                             vec![5,1,8,8,4,6,8,5,2,4,1,6,2,2,9,7]];
        let big_grid_result = 83;
        let result = leetcode_64(vec![vec![1,3,1],vec![1,5,1],vec![4,2,1]]);
        assert_eq!(result, 7);
        let result = leetcode_64(vec![vec![1,2,3],vec![4,5,6]]);
        assert_eq!(result, 12);
        let result = leetcode_64_iterative(vec![vec![1,3,1],vec![1,5,1],vec![4,2,1]]);
        assert_eq!(result, 7);
        let result = leetcode_64_iterative(vec![vec![1,2,3],vec![4,5,6]]);
        assert_eq!(result, 12);
        let result = leetcode_64_iterative(big_grid);
        assert_eq!(result, big_grid_result);
    }

    #[test]
    fn test_leetcode_62() {
        let result = leetcode_62(3,7);
        assert_eq!(result, 28);
        let result = leetcode_62(3,2);
        assert_eq!(result, 3);
        let result = leetcode_62_recursive(3,7);
        assert_eq!(result, 28);
        let result = leetcode_62_recursive(3,2);
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
