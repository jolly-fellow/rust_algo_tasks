use std::cmp;
use std::cmp::max;
use std::cmp::min;
use std::collections::HashMap;
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
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        return 1;
    }
    leetcode_1137(n - 1) + leetcode_1137(n - 2) + leetcode_1137(n - 3)
}

// 1137. N-th Tribonacci Number iterative solution
pub fn leetcode_1137_iterative(n: i32) -> i32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    if n == 2 {
        return 1;
    }
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
    if v.len() == 1 {
        return v[0];
    }
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

    let sum = n
        .iter()
        .skip(2)
        .fold((n[0], n[1]), |(prev2, prev1), &current| {
            (prev1, max(current + prev2, prev1))
        })
        .1;

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
            g[row][col] + min(r(g, row - 1, col), r(g, row, col - 1))
        }
    }
    r(&grid, grid.len() - 1, grid[0].len() - 1)
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
    fn r(m: i32, n: i32, g: &Vec<Vec<i32>>) -> i32 {
        if m > g.len() as i32 - 1
            || n > g[0].len() as i32 - 1
            || m < 0
            || n < 0
            || g[m as usize][n as usize] == 1
        {
            0
        } else if m == g.len() as i32 - 1 && n == g[0].len() as i32 - 1 {
            1
        } else {
            r(m + 1, n, g) + r(m, n + 1, g)
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
            grid[i][j] += min(
                grid[i - 1][j],
                min(
                    grid[i - 1][j.saturating_sub(1)],
                    grid[i - 1][min(cols - 1, j + 1)],
                ),
            );
        }
    }
    *grid.last().unwrap().iter().min().unwrap()
}

// https://leetcode.com/problems/maximal-square/
// 221. Maximal Square
pub fn leetcode_221(matrix: &Vec<Vec<char>>) -> i32 {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut dp = vec![vec![0; cols + 1]; rows + 1];
    let mut max_side = 0;

    for r in 0..rows {
        for c in 0..cols {
            if matrix[r][c] == '1' {
                dp[r + 1][c + 1] = dp[r][c].min(dp[r + 1][c]).min(dp[r][c + 1]) + 1;
                max_side = max_side.max(dp[r + 1][c + 1]);
            }
        }
    }
    max_side * max_side
}

// optimized solution by memory size using a vector for dp instead of matrix because
// we don't need to keep processed lines of the given matrix.

pub fn leetcode_221_vector(matrix: &Vec<Vec<char>>) -> i32 {
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
pub fn leetcode_221_vector_iter(matrix: &Vec<Vec<char>>) -> i32 {
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
    if s.len() <= 1 {
        return s;
    }

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

// https://leetcode.com/problems/longest-palindromic-subsequence
// 516. Longest Palindromic Subsequence
pub fn leetcode_516_recursive(s: String) -> i32 {
    // Simple recursive solution
    fn r(s: &Vec<char>, b: isize, e: isize) -> i32 {
        if b == e {
            1
        } else if b > e {
            0
        } else if s[b as usize] == s[e as usize] {
            2 + r(s, b + 1, e - 1)
        } else {
            max(r(s, b + 1, e), r(s, b, e - 1))
        }
    }
    r(&s.chars().collect(), 0, s.len() as isize - 1)
}

// Dynamic programming solution
pub fn leetcode_516_dp(s: String) -> i32 {
    let n = s.len();
    let s_chars: Vec<char> = s.chars().collect();
    let mut dp = vec![vec![0; n]; n];

    // Initialize diagonal elements because one symbol length
    // sequence is always a palindrome
    for i in 0..n {
        dp[i][i] = 1;
    }

    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            if s_chars[i] == s_chars[j] {
                dp[i][j] = dp[i + 1][j - 1] + 2;
            } else {
                dp[i][j] = dp[i + 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[0][n - 1]
}

// https://leetcode.com/problems/edit-distance/
// 72. Edit Distance
pub fn leetcode_72(s1: String, s2: String) -> usize {
    let m = s1.len();
    let n = s2.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp: Vec<Vec<usize>> = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        dp[i][0] = i;
    }
    for j in 1..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            if s1.chars().nth(i - 1) == s2.chars().nth(j - 1) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = usize::min(
                    dp[i - 1][j - 1] + 1,
                    usize::min(dp[i][j - 1] + 1, dp[i - 1][j] + 1),
                );
            }
        }
    }
    dp[m][n]
}

// Used dp array instead of dp matrix.
// Speed of work with chars by index was improved
pub fn leetcode_72_opt(s1: String, s2: String) -> i32 {
    let m = s1.len();
    let n = s2.len();

    // Early return for empty strings
    if m == 0 {
        return n as i32;
    }
    if n == 0 {
        return m as i32;
    }

    // Use slices to avoid copying strings
    let s1_bytes = s1.as_bytes();
    let s2_bytes = s2.as_bytes();

    // Pre-allocate array for intermediate results
    let mut costs = vec![0; n + 1];

    // Initialize the first row
    for j in 1..=n {
        costs[j] = j;
    }

    // Iterate through the remaining rows
    for i in 1..=m {
        let mut prev = costs[0];
        costs[0] = i;

        for j in 1..=n {
            let cost = if s1_bytes[i - 1] == s2_bytes[j - 1] {
                prev
            } else {
                usize::min(prev + 1, usize::min(costs[j] + 1, costs[j - 1] + 1))
            };
            prev = costs[j];
            costs[j] = cost;
        }
    }
    costs[n] as i32
}

// https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings
// 712. Minimum ASCII Delete Sum for Two Strings
// Recursive solution
/*
pub fn leetcode_712(s1: String, s2: String) -> i32 {
    fn r(s1: &[u8], i1: usize, s2: &[u8], i2: usize) -> isize {
        let s1_end = i1 == s1.len();
        let s2_end = i2 == s2.len();

        if s1_end || s2_end {
            return if s1_end {
                s2.iter().skip(i2).map(|&c| c as isize).sum()
            }
            else {
                s1.iter().skip(i1).map(|&c| c as isize).sum()
            }
        }

        if s1[i1] == s2[i2] {
            r(s1, i1+1, s2, i2+1)
        }
        else {
            min(r(s1, i1+1, s2, i2) + s1[i1] as isize,
                r(s1, i1, s2, i2+1) + s2[i2] as isize)
        }
    }
    r(s1.as_bytes(), 0, s2.as_bytes(), 0) as i32
}
*/

// Recursive solution with memoization in a matrix
// it is a good example which shows that the dynamic programming is
// just a kind of memoization. The matrix may be changed to a classic hash table but it will
// works slower

pub fn leetcode_712(s1: String, s2: String) -> i32 {
    fn r(s1: &[u8], i1: usize, s2: &[u8], i2: usize, matrix: &mut Vec<Vec<isize>>) -> isize {
        let s1_end = i1 == s1.len();
        let s2_end = i2 == s2.len();

        if s1_end || s2_end {
            return if s1_end {
                s2.iter().skip(i2).map(|&c| c as isize).sum()
            } else {
                s1.iter().skip(i1).map(|&c| c as isize).sum()
            };
        }

        if matrix[i1][i2] != -1 {
            return matrix[i1][i2];
        }

        let sum = if s1[i1] == s2[i2] {
            r(s1, i1 + 1, s2, i2 + 1, matrix)
        } else {
            min(
                r(s1, i1 + 1, s2, i2, matrix) + s1[i1] as isize,
                r(s1, i1, s2, i2 + 1, matrix) + s2[i2] as isize,
            )
        };
        matrix[i1][i2] = sum;
        sum
    }
    let mut matrix: Vec<Vec<isize>> = vec![vec![-1; s2.len() + 1]; s1.len() + 1];
    r(s1.as_bytes(), 0, s2.as_bytes(), 0, &mut matrix) as i32
}

// Real DP solution
pub fn leetcode_712_dp(s1: String, s2: String) -> i32 {
    let s1_len = s1.len();
    let s2_len = s2.len();
    let b1 = s1.as_bytes();
    let b2 = s2.as_bytes();
    let mut m: Vec<Vec<i32>> = vec![vec![0; s2_len + 1]; s1_len + 1];

    for j in 1..=s2_len {
        m[0][j] = m[0][j - 1] + b2[j - 1] as i32;
    }

    for i in 1..=s1_len {
        m[i][0] = m[i - 1][0] + b1[i - 1] as i32;
        for j in 1..=s2_len {
            if b1[i - 1] == b2[j - 1] {
                m[i][j] = m[i - 1][j - 1];
            } else {
                m[i][j] = i32::min(
                    m[i - 1][j] + b1[i - 1] as i32,
                    m[i][j - 1] + b2[j - 1] as i32,
                );
            }
        }
    }
    m[s1_len][s2_len]
}

// Used array instead of the matrix
pub fn leetcode_712_dp_opt(s1: String, s2: String) -> i32 {
    let s1_len = s1.len();
    let s2_len = s2.len();
    let b1 = s1.as_bytes();
    let b2 = s2.as_bytes();
    let mut dp: Vec<i32> = vec![0; s2_len + 1];

    for j in 1..=s2_len {
        dp[j] = dp[j - 1] + b2[j - 1] as i32;
    }

    for i in 1..=s1_len {
        let mut tmp1 = dp[0];
        dp[0] += b1[i - 1] as i32;

        for j in 1..=s2_len {
            let tmp2 = dp[j];
            dp[j] = if b1[i - 1] == b2[j - 1] {
                tmp1
            } else {
                min(dp[j] + b1[i - 1] as i32, dp[j - 1] + b2[j - 1] as i32)
            };
            tmp1 = tmp2;
        }
    }
    dp[s2_len]
}

// https://leetcode.com/problems/distinct-subsequences/description/
// 115. Distinct Subsequences
pub fn leetcode_115(s: &str, t: &str) -> i32 {
    // This function counts the number of subsequences of the string `s` that are also subsequences of the string `t`.
    // A subsequence is a string that can be formed by deleting some (but not all) of the characters from another string,
    // without changing the order of the remaining characters. For example, "abc" is a subsequence of "ahbgdc",
    // but "acb" is not.

    // Base cases:
    // If the target string `t` is empty, then any string `s` is a subsequence of it, so we return 1.
    if t.is_empty() {
        return 1;
    }
    // If the source string `s` is empty, then there are no subsequences of it, so we return 0.
    if s.is_empty() {
        return 0;
    }

    // Handle characters:
    // Get the first character of each string.
    let curr_s_char = s.chars().next().unwrap();
    let curr_t_char = t.chars().next().unwrap();

    // **Include current character only if it matches:**
    // If the first character of the source string `s` matches the first character of the target string `t`,
    // then there are two possibilities:
    // 1. Include the current character in the subsequence. In this case, we recursively call the function with
    // the remaining characters of both strings (`s[1..]` and `t[1..]`).
    // 2. Exclude the current character from the subsequence. In this case, we recursively call the function with
    // the remaining characters of the source string `s[1..]` and the original target string `t`.
    // We only count the subsequence if the current character matches, so we set `count_with_char` to 0 otherwise.
    let count_with_char = if curr_s_char == curr_t_char {
        // Recursively call with both possibilities:
        leetcode_115(&s[1..], &t[1..]) // Include character
    } else {
        0 // Don't include if characters don't match
    };

    // **Exclude current character (always check the original t!):**
    // Even if the current character does not match, we still need to consider the possibility of excluding it from the subsequence.
    // In this case, we recursively call the function with the remaining characters of the source string `s[1..]` and the original target string `t`.
    let count_without_char = leetcode_115(&s[1..], t);

    // Combine counts:
    // The total number of subsequences is the sum of the number of subsequences that include the current character
    // and the number of subsequences that exclude the current character.
    count_with_char + count_without_char
}

pub fn leetcode_115_dp(s1: &str, s2: &str) -> i32 {
    let s = s1.as_bytes(); // source string
    let t = s2.as_bytes(); // target string
    let s_len = s.len();
    let t_len = t.len();

    // Create a 2D dynamic programming table with size (t_len+1) x (s_lenn+1)
    // This table will store the number of distinct subsequences of `t[0..i-1]` in `s[0..j-1]`
    let mut dp = vec![vec![0; s_len + 1]; t_len + 1];

    // Base cases:
    // - An empty string is a subsequence of any string, so initialize the first row to 1.
    for i in 0..=s_len {
        dp[0][i] = 1;
    }

    // Iterate through the dynamic programming table, starting from i = 1 (second character of t)
    // and j = 1 (second character of s)
    for i in 1..=t_len {
        for j in 1..=s_len {
            // Check if the characters at the current positions match
            if s[j - 1] == t[i - 1] {
                // If they match, there are two options:
                // 1. Include the current character: count = dp[i - 1][j - 1]
                // (consider subsequence ending at previous characters)
                // 2. Exclude the current character: count = dp[i][j - 1]
                // (consider subsequence excluding the previous character)
                dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1];
            } else {
                // If they don't match, the only option is to exclude the current character:
                // count = dp[i][j - 1]
                dp[i][j] = dp[i][j - 1];
            }
        }
    }
    // The answer is the number of distinct subsequences of `t` in `s`,
    // located at the bottom right corner of the table
    dp[t_len][s_len]
}

// Used an array instead of a matric
pub fn leetcode_115_dp_opt(s1: &str, s2: &str) -> i32 {
    let t = s2.as_bytes();
    let s = s1.as_bytes();
    let mut dp = vec![0; t.len() + 1];

    // Initialize base cases
    dp[0] = 1; // Empty string is always a subsequence

    for c in s {
        for j in (0..t.len()).rev() {
            let prev = dp[j + 1];
            dp[j + 1] = if *c == t[j] { dp[j] + prev } else { prev };
        }
    }
    dp[t.len()]
}

// https://leetcode.com/problems/longest-increasing-subsequence
// 300. Longest Increasing Subsequence
pub fn leetcode_300_recursive(v: Vec<i32>) -> i32 {
    fn r(v: &Vec<i32>, i: usize, prev: i32) -> i32 {
        if i >= v.len() {
            return 0;
        }
        let mut take = 0;
        // check ways to build the sequence without current element v[i]
        let dont_take = r(v, i + 1, prev);
        // add the current element v[i] to the sequence if it is greater than the previous one
        if v[i] > prev {
            take = 1 + r(v, i + 1, v[i]);
        }
        // compare which sequence is better with or without the current element
        max(take, dont_take)
    }
    r(&v, 0, i32::MIN)
}

pub fn leetcode_300_iterative(v: Vec<i32>) -> i32 {
    let mut dp = vec![1; v.len()];
    let mut result = 1;

    for i in 0..v.len() {
        for j in 0..i {
            if v[i] > v[j] {
                dp[i] = max(dp[i], dp[j] + 1);
                result = max(result, dp[i]);
            }
        }
    }
    result
}

// 673. Number of Longest Increasing Subsequence
// https://leetcode.com/problems/number-of-longest-increasing-subsequence

pub fn leetcode_673_iterative(v: Vec<i32>) -> i32 {
    let mut dp: Vec<(i32, i32)> = vec![(1, 1); v.len()];
    let mut result = 0;
    let mut max_len = 0;

    for i in 0..v.len() {
        for j in 0..i {
            if v[i] > v[j] {
                if dp[i].0 == dp[j].0 + 1 {
                    dp[i].1 += dp[j].1
                }
                if dp[i].0 < dp[j].0 + 1 {
                    dp[i] = (dp[j].0 + 1, dp[j].1)
                }
            }
        }
        if max_len == dp[i].0 {
            result += dp[i].1
        }
        if max_len < dp[i].0 {
            max_len = dp[i].0;
            result = dp[i].1;
        }
    }
    result
}

// 646. Maximum Length of Pair Chain
// https://leetcode.com/problems/maximum-length-of-pair-chain

pub fn leetcode_646_iterative(mut vec: Vec<Vec<i32>>) -> i32 {
    vec.sort_by(|a, b| a[1].cmp(&b[1]));
    let mut prev = 0;
    let mut res = 1;

    for i in 1..vec.len() {
        if vec[prev][1] < vec[i][0] {
            prev = i;
            res += 1;
        }
    }
    res
}

// 1218. Longest Arithmetic Subsequence of Given Difference
// https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/
pub fn leetcode_1218_iterative(v: Vec<i32>, diff: i32) -> i32 {
    let mut dp = HashMap::new();
    let mut longest = 0;

    for &i in v.iter() {
        let prev_length = *dp.get(&(i - diff)).unwrap_or(&0);
        let curr_length = prev_length + 1;
        dp.insert(i, curr_length);
        longest = longest.max(curr_length);
    }
    longest
}

// 1027. Longest Arithmetic Subsequence
// https://leetcode.com/problems/longest-arithmetic-subsequence

pub fn leetcode_1027_iterative(v: Vec<i32>) -> i32 {
    let n = v.len();
    let mut dp = vec![HashMap::new(); n];
    let mut max_len = 2;

    for i in 0..n {
        for j in 0..i {
            let diff = v[i] - v[j];
            let len = *dp[j].get(&diff).unwrap_or(&1) + 1;
            dp[i].insert(diff, len);
            max_len = max_len.max(len);
        }
    }
    max_len
}

// 354. Nested Envelopes
// https://leetcode.com/problems/russian-doll-envelopes/

// Classic DP. Slow O(n^2)
/*
pub fn max_envelopes(envelopes: Vec<Vec<i32>>) -> i32 {
    let mut envelopes = envelopes;
    envelopes.sort_by(|a, b| {
        if a[0] == b[0] {
            b[1].cmp(&a[1])
        } else {
            a[0].cmp(&b[0])
        }
    });

    let mut dp = vec![1; envelopes.len()];
    let mut max_count = 1;

    for i in 1..envelopes.len() {
        for j in 0..i {
            if envelopes[i][1] > envelopes[j][1] {
                dp[i] = std::cmp::max(dp[i], dp[j] + 1);
            }
        }
        max_count = std::cmp::max(max_count, dp[i]);
    }

    max_count
}
*/

/*
We can use a more efficient algorithm that has a time complexity of O(n log n).
One such algorithm is the Patience sorting algorithm, which is a variant of the
Longest Increasing Subsequence (LIS) problem. The idea is to create a pile of cards where each
pile is a subsequence that we can extend by adding a new card.

In this function, we first sort the envelopes by width in ascending order and then by height in
descending order. This is done using the sort_unstable_by_key function, which sorts the envelopes in-place.

Then, we create a tails vector to store the top card of each pile. For each envelope, we use binary
search to find the position where we can place the envelope's height. If the height is larger than
all the piles, we start a new pile. Otherwise, we replace the top card of the pile with the smaller height.

Finally, the length of tails is the number of piles, which is the maximum number of envelopes that
can be Russian dolled.

This algorithm has a time complexity of O(n log n) because of the sorting step, and a space complexity
of O(n) for the tails vector. This is much more efficient than the previous approach for large inputs.
 */
pub fn leetcode_354_iterative(envelopes: Vec<Vec<i32>>) -> i32 {
    let mut envelopes = envelopes;
    envelopes.sort_unstable_by_key(|k| (k[0], -k[1]));

    let mut tails = Vec::new();

    for envelope in envelopes {
        let height = envelope[1];
        if let Err(idx) = tails.binary_search(&height) {
            if idx == tails.len() {
                tails.push(height);
            } else {
                tails[idx] = height;
            }
        }
    }
    tails.len() as i32
}

// 1964. Find the Longest Valid Obstacle Course at Each Position
// https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/

/*
Algorithm description:
Certainly, let's break down the algorithm:

1. **Initialize Variables**: We start by initializing three vectors: `dp`, `res`, and `dp_index`.
`dp` is used to store the longest increasing subsequence found so far. `res` is used to store the
result for each index, which represents the length of the longest obstacle course for each index.
`dp_index` is used to keep track of the indices of the elements in the `dp` vector.

2. **Iterate Over Obstacles**: We then iterate over each obstacle in the `obstacles` vector.
For each obstacle, we perform the following steps:

   a. **Find Upper Bound**: We use a binary search to find the position where the current obstacle
   should be inserted in the `dp` vector. This is done by calling the `upper_bound` function,
   which returns the index of the first element in `dp` that is greater than the current obstacle.

   b. **Update `dp` and `dp_index`**: If the upper bound is equal to the current length of `dp`,
   it means we have found a new, longer increasing subsequence. We append the current obstacle
   to `dp` and its index to `dp_index`.

   c. **Update `dp` and `dp_index`**: If the upper bound is less than the current length of `dp`,
   it means we have found a shorter increasing subsequence for the current obstacle.
   We update the element at the upper bound position in `dp` and its index in `dp_index`.

   d. **Update `res`**: We update the result for the current index in `res`.
   If the upper bound is equal to the current length of `dp`, it means we have found a new,
   longer increasing subsequence, so we add 1 to the upper bound to get the length of the longest
   obstacle course. If the upper bound is less than the current length of `dp`, it means we have
   found a shorter increasing subsequence, so we use the upper bound as the length of the longest
   obstacle course.

3. **Return Result**: After iterating over all obstacles, we return the `res` vector, which contains
the length of the longest obstacle course for each index.

The time complexity of this algorithm is O(n log n) because for each obstacle, we perform a binary
search in the `dp` vector, which takes O(log n) time. The space complexity is O(n) because we
use three vectors of size n.
*/

pub fn leetcode_1964_iterative(obstacles: Vec<i32>) -> Vec<i32> {
    fn upper_bound(dp: &[i32], left: usize, right: usize, target: i32) -> usize {
        let mut left = left;
        let mut right = right;

        while left < right {
            let mid = left + (right - left) / 2;
            if dp[mid] <= target {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }

    let n = obstacles.len();
    let mut dp = vec![0; n];
    let mut res = vec![0; n];
    let mut len = 0;

    for i in 0..n {
        let j = upper_bound(&dp, 0, len, obstacles[i]);
        if j == len {
            len += 1;
        }
        dp[j] = obstacles[i];
        res[i] = (j + 1) as i32;
    }
    res
}

// 1143. Longest Common Subsequence
// https://leetcode.com/problems/longest-common-subsequence/

/*
 LCS is a problem of finding the longest subsequence common to all sequences in a set of sequences
 (often just two sequences). A subsequence is a sequence that appears in the same relative order,
 but not necessarily contiguous.

Here's a step-by-step explanation of the algorithm:

    Initialize a 2D Dynamic Programming Table (DP Table):
        The DP table is a 2D array where dp[i][j] represents the length of the longest common
        subsequence for the first i characters of text1 and the first j characters of text2.
        The table is initialized with zeros, with an extra row and column for the base case
        (when one of the strings is empty).

    Fill the DP Table:
        Iterate over each character in text1 and text2.
        For each pair of characters at indices i and j in text1 and text2, respectively:
            If the characters are the same (text1[i - 1] == text2[j - 1]), then the current cell
            dp[i][j] is equal to the cell diagonally above it plus one (dp[i - 1][j - 1] + 1).
            This represents the common subsequence that includes the current character.

            If the characters are different, then the current cell dp[i][j] is equal to the maximum
            of the cell to the left (dp[i][j - 1]) and the cell above (dp[i - 1][j]).
            This represents the maximum length of the common subsequence found so far without
            the current character.

    Return the Value from the Bottom-Right Cell of the DP Table:
        The bottom-right cell dp[m][n] of the DP table contains the length of the longest common
        subsequence of the entire text1 and text2.
*/


pub fn leetcode_1143_iterative(text1: String, text2: String) -> i32 {
    // Convert the strings to byte slices for efficient memory access
    let text1: &[u8] = text1.as_bytes();
    let text2: &[u8] = text2.as_bytes();

    // Get the lengths of the byte slices
    let (m, n) = (text1.len(), text2.len());

    // Initialize the DP table with an extra row and column for the base case
    let mut dp = vec![vec![0; n + 1]; m + 1];

    // Fill the DP table
    for i in 1..=m {
        for j in 1..=n {
            // If the current characters are the same, increment the length of the LCS
            if text1[i - 1] == text2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                // If the characters are different, take the maximum LCS length so far
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    // The bottom-right cell contains the length of the LCS for the entire strings
    // Convert the integer to i32 and return
    dp[m][n]
}

// Optimized memory using
pub fn leetcode_1143_opt(text1: String, text2: String) -> i32 {
    let text1: &[u8] = text1.as_bytes();
    let text2: &[u8] = text2.as_bytes();
    let mut dp = vec![0; text2.len() + 1];
    let mut prev_dp_val; // Variable to hold the previous diagonal value

    for &c1 in text1 {
        prev_dp_val = 0; // Reset prev_dp_val for each new character in text1
        for (j, &c2) in text2.iter().enumerate() {
            let temp = dp[j + 1]; // Store the current dp[j+1] before it's updated
            if c1 == c2 {
                dp[j + 1] = prev_dp_val + 1; // Use prev_dp_val to update dp[j+1]
            } else {
                dp[j + 1] = dp[j + 1].max(dp[j]); // Update dp[j+1] based on the max of dp[j+1] and dp[j]
            }
            prev_dp_val = temp; // Update prev_dp_val to the value of dp[j+1] before it was updated
        }
    }
    dp[text2.len()]
}

// 1035. Uncrossed Lines
// https://leetcode.com/problems/uncrossed-lines
// The task actually is the same as the previous task because we can draw parallel lines only between
// numbers in common sequences so solution is the same too:

pub fn leetcode_1035_iterative(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let (m, n) = (nums1.len(), nums2.len());
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if nums1[i - 1] == nums2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[m][n]
}

// 1312. Minimum Insertion Steps to Make a String Palindrome
// https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/
/*
This function uses dynamic programming to find the length of the longest palindromic subsequence
(LPS) of the given string `s`. The LPS is the longest subsequence of the string that is also a palindrome.
By finding the LPS, we can determine the minimum number of characters that need to be inserted to
make the string a palindrome. The minimum number of insertions is the length of the string minus the length of the LPS.

The function first initializes a 2D DP table `dp` with all elements set to `0`. It then fills in the
DP table by considering all substrings of increasing length starting from length `2` up to the entire
string `s`. For each substring, it checks if the characters at the start and end are the same. If they
are, it means we can include both of them in the LPS, and the length is `2` plus the length of the LPS
in the middle substring `s[(i+1)..=(j-1)]`. If the characters are not the same, we take the maximum length
of the LPS we can get by excluding either the first or the last character.

Finally, the function returns the difference between the length of the string `s` and the length of the
LPS for the entire string `s[0..=(n-1)]`, which represents the minimum number of characters that need
to be inserted to make `s` a palindrome.
*/
pub fn leetcode_1312_iterative(s: String) -> i32 {
    // Convert the input string `s` into a byte array `&[u8]`.
    // This is done to avoid creating a new vector of characters, which can be memory-intensive.
    let s: &[u8] = s.as_bytes();
    // Get the length of the string `s`.
    let n = s.len();
    // Initialize a 2D dynamic programming (DP) table `dp` with dimensions `n x n`.
    // Each element `dp[i][j]` in the DP table will represent the length of the longest
    // palindromic subsequence (LPS) in the substring `s[i..=j]`.
    let mut dp = vec![vec![0; n]; n];
    // Initialize the DP table for single character substrings.
    // A single character is always a palindrome of length `1`.
    for i in 0..n {
        dp[i][i] = 1;
    }
    // Fill the DP table for substrings of length 2 to n.
    for len in 2..=n {
        for i in 0..=(n - len) {
            let j = i + len - 1;
            // If the characters at the start and end of the substring `s[i]` and `s[j]` are the same,
            // then the length of the LPS in `s[i..=j]` is `2` plus the length of the LPS in the
            // substring `s[(i+1)..=(j-1)]`.
            if s[i] == s[j] {
                dp[i][j] = 2 + dp[i + 1][j - 1];
            } else {
                // If the characters are not the same, then the length of the LPS in `s[i..=j]` is
                // the maximum of the LPS in the substring `s[(i+1)..=j]` and the LPS in the
                // substring `s[i..=(j-1)]`.
                dp[i][j] = dp[i + 1][j].max(dp[i][j - 1]);
            }
        }
    }
    // The minimum number of insertions is the difference between the length of `s` and the length
    // of the LPS for the entire string `s[0..=(n-1)]`, which is `dp[0][n-1]`.
    (n - dp[0][n - 1]) as i32
}


// 309. Best Time to Buy and Sell Stock with Cooldown
// https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
/*
Classic Dynamic Programming approach.
The idea is to keep track of the maximum profit that can be achieved at each step, considering the
current state of the stock (owned or not owned), the number of transactions made, and the day of the
trading.

Here's a detailed description of the algorithm:

    Initialize a 3D array dp where dp[i][j][k] represents the maximum profit that can be achieved
    on the i-th day with j transactions and k stocks in hand (k is either 0 or 1).

    Set the base cases:
        dp[0][0][0] = 0: No profit on the first day with 0 transactions and no stocks in hand.
        dp[0][0][1] = -prices[0]: Buying the stock on the first day will result in a loss of the
        price of the stock.

    Iterate over the days of trading (from 1 to n), and for each day, iterate over the number of
    transactions allowed (from 0 to 2).

    For each day and transaction count, update the maximum profit considering two scenarios:

        If we don't have a stock on the current day (k = 0), we can either do nothing
        (dp[i-1][j][0]) or sell the stock we have (dp[i-1][j-1][1] + prices[i-1]).
        If we have a stock on the current day (k = 1), we can either do nothing
        (dp[i-1][j][1]) or buy a stock (dp[i-2][j][0] - prices[i-1]).
        Note that we can't buy a stock on the same day we sold a stock, so we use
        dp[i-2][j][0] to account for the cooldown period.

    After filling the dp array, return the maximum profit that can be achieved with at most 2
    transactions and no stocks in hand on the last day (dp[n][2][0]).
 */
pub fn leetcode_309_iterative(prices: Vec<i32>) -> i32 {
    let n = prices.len();
    // Initialize the dp array with zeros
    let mut dp = vec![vec![vec![0; 2]; 3]; n + 1];
    // Set the base cases
    dp[0][0][1] = i32::MIN; // Impossible to have a stock on day 0
    dp[0][1][1] = i32::MIN; // Impossible to have a stock on day 0
    dp[0][2][1] = i32::MIN; // Impossible to have a stock on day 0

    // Iterate over the days
    for i in 1..=n {
        // Iterate over the number of transactions allowed
        for j in 0..=2 {
            // If we don't have a stock on the current day
            dp[i][j][0] = dp[i-1][j][0]; // Do nothing
            if j > 0 {
                // Sell the stock we have
                dp[i][j][0] = dp[i][j][0].max(dp[i-1][j-1][1] + prices[i-1]);
            }
            // If we have a stock on the current day
            dp[i][j][1] = dp[i-1][j][1]; // Do nothing
            if i > 1 {
                // Buy a stock (with cooldown)
                dp[i][j][1] = dp[i][j][1].max(dp[i-2][j][0] - prices[i-1]);
            } else {
                // First day, we can only buy a stock
                dp[i][j][1] = dp[i][j][1].max(-prices[i-1]);
            }
        }
    }
    // Return the maximum profit with at most 2 transactions and no stocks in hand on the last day
    dp[n][2][0]
}
/*
Optimized version. It uses a more space-efficient approach by only keeping track of two arrays
(buy and sell) instead of a 3D array. This reduces the space complexity from O(n) to O(1),
which can be beneficial when dealing with large inputs.

Algorithm Description

The algorithm is designed to find the maximum profit that can be achieved by buying and selling a
stock with a cooldown period. The cooldown period means that after selling a stock, you cannot buy
stock on the next day. The algorithm uses dynamic programming to keep track of the maximum profit
that can be achieved at each step.

Here's a step-by-step description of the algorithm:

    Initialize two arrays, buy and sell, with the same length as the prices array. These arrays will
    store the maximum profit that can be achieved at each step.

    Set the base cases for the buy and sell arrays. The buy array is initialized with the negative
    of the first price since buying stock on the first day will result in a loss. The sell array is
    initialized with 0 because no profit can be made on the first day.

    Iterate through the prices array starting from the second day (index 1). For each day, update
    the buy and sell arrays as follows:

        To calculate the maximum profit if we buy a stock on the i-th day, we consider two options:

            We don't buy a stock on the i-th day, so the profit is the same as the previous day
            (buy[i - 1]).

            We sell a stock on the day before the i-th day (sell[i - 2]) and then buy a stock
            on the i-th day, which results in a loss of prices[i].

            We choose the maximum of these two options.

        To calculate the maximum profit if we sell a stock on the i-th day, we consider two options:

            We don't sell a stock on the i-th day, so the profit is the same as the previous day
            (sell[i - 1]).

            We buy a stock on the i-th day (buy[i - 1]) and then sell it on the i-th day, which
            results in a gain of prices[i].

            We choose the maximum of these two options.

    After iterating through all the days, the maximum profit is stored in sell[n - 1],
    where n is the number of days.
*/
pub fn leetcode_309_opt(prices: Vec<i32>) -> i32 {
    let n = prices.len();
    if n < 2 {
        return 0; // If there's no price data or only one day, no profit can be made.
    }

    // Initialize two arrays to keep track of the maximum profit.
    let mut buy = vec![0; n];
    let mut sell = vec![0; n];

    // Base cases for the first two days.
    buy[0] = -prices[0]; // Buying stock on the first day.
    buy[1] = std::cmp::max(-prices[0], -prices[1]); // Buying stock on the second day.
    sell[1] = std::cmp::max(0, prices[1] - prices[0]); // Selling stock on the second day.

    // Iterate through the rest of the days.
    for i in 2..n {
        // Update the maximum profit if we buy a stock on the i-th day.
        buy[i] = std::cmp::max(buy[i - 1], sell[i - 2] - prices[i]);
        // Update the maximum profit if we sell a stock on the i-th day.
        sell[i] = std::cmp::max(sell[i - 1], buy[i - 1] + prices[i]);
    }
    // The maximum profit is in the last element of the sell array.
    sell[n - 1]
}

// 714. Best Time to Buy and Sell Stock with Transaction Fee
// https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee
/*
The problem is about maximizing the profit from buying and selling stocks, with a transaction fee
that is incurred on each transaction. We can make as many transactions as we want, but we have to
pay the fee for each one.

The algorithm uses dynamic programming to keep track of two states:

1. `sell`: The maximum profit we can make if we're not holding any stock.
2. `buy`: The maximum profit we can make if we're currently holding a stock.

Initially, we start with `sell = 0` because we haven't made any profit yet, and `buy = -prices[0]`
because we're buying the first stock, so our profit (or loss) is the negative of the price of the
first stock.

Now, for each day starting from the second day, we update `sell` and `buy` as follows:

- If we decide to sell the stock today, we would have made a profit equal to the price of the stock
on that day, minus the transaction fee. But we can only sell if we had bought the stock before, so
we add the `buy` state to this profit. We then compare this with the current `sell` state to see if
selling today would be more profitable than not selling.

- If we decide to buy the stock today, we would have spent the price of the stock on that day.
But we can only buy if we sold the stock before, so we subtract the `sell` state from this cost.
We then compare this with the current `buy` state to see if buying today would be more profitable
than not buying.

After going through all the days, `sell` will hold the maximum profit we can make, because we can't
hold a stock after the last day.

This algorithm is efficient because it only requires a single pass through the prices, and it only
uses a constant amount of space.
 */
pub fn leetcode_714(prices: Vec<i32>, fee: i32) -> i32 {
    // Initialize the maximum profit if we don't have a stock (sell) and the maximum profit if we do have a stock (buy)
    // We start with 0 profit if we don't have a stock and -prices[0] profit if we buy the first stock
    let mut sell = 0;
    let mut buy = -prices[0];

    // Iterate over the prices starting from the second day
    for &price in prices.iter().skip(1) {
        // If we decide to sell the stock today, the profit is the previous buy state (buying at the min price)
        // plus the current price minus the transaction fee
        sell = sell.max(buy + price - fee);
        // If we decide to buy the stock today, the profit is the previous sell state (selling at the max price)
        // minus the current price
        buy = buy.max(sell - price);
    }
    // Return the maximum profit we can make if we end up with no stock
    sell
}

// Functional style solution
pub fn leetcode_714_func(prices: Vec<i32>, fee: i32) -> i32 {
    prices.iter().fold((0, -prices[0]), |(sell, buy), price| {
        (sell.max(buy + price - fee), buy.max(sell - price))
    }).0
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! vec_of_strings {
        ($($x:expr),*) => (vec![$($x.to_string()),*]);
    }

    fn create_vectors() -> Vec<Vec<i32>> {
        (1..=100_000).map(|i| vec![i, i]).collect()
    }

    /*
    macro_rules! create_vectors {
        () => {
            (1..=100_000).map(|i| vec![i, i]).collect()
        };
    }

        fn create_vectors() -> Vec<Vec<i32>> {
            (1..=100_000).map(|i| vec![i, i]).collect()
        }

        #[test]
        fn test_leetcode_() {
            let result = leetcode_();
            assert_eq!(result, );
        }
    */

    #[test]
    fn test_leetcode_714() {
        let result = leetcode_714(vec![1,3,2,8,4,9], 2);
        assert_eq!(result, 8);
        let result = leetcode_714(vec![1,3,7,5,10,3], 3);
        assert_eq!(result, 6);
        let result = leetcode_714_func(vec![1,3,2,8,4,9], 2);
        assert_eq!(result, 8);
        let result = leetcode_714_func(vec![1,3,7,5,10,3], 3);
        assert_eq!(result, 6);

    }

    #[test]
    fn test_leetcode_309() {
        let result = leetcode_309_iterative(vec![1,2,3,0,2]);
        assert_eq!(result, 3);
        let result = leetcode_309_iterative(vec![1]);
        assert_eq!(result, 0);
        let result = leetcode_309_opt(vec![1,2,3,0,2]);
        assert_eq!(result, 3);
        let result = leetcode_309_opt(vec![1]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_leetcode_1312() {
        let result = leetcode_1312_iterative("zzazz".to_string());
        assert_eq!(result, 0);
        let result = leetcode_1312_iterative("mbadm".to_string());
        assert_eq!(result, 2);
        let result = leetcode_1312_iterative("leetcode".to_string());
        assert_eq!(result, 5);
    }

    #[test]
    fn test_leetcode_1035() {
        let result = leetcode_1035_iterative(vec![1,3,7,1,7,5], vec![1,9,2,5,1]);
        assert_eq!(result, 2);
        let result = leetcode_1035_iterative(vec![1,4,2], vec![1,2,4]);
        assert_eq!(result, 2);
        let result = leetcode_1035_iterative(vec![2,5,1,2,5], vec![10,5,2,1,5,2]);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_leetcode_1143() {
        let result = leetcode_1143_iterative("abcde".to_string(), "ace".to_string());
        assert_eq!(result, 3);
        let result = leetcode_1143_iterative("abc".to_string(), "abc".to_string());
        assert_eq!(result, 3);
        let result = leetcode_1143_iterative("abc".to_string(), "def".to_string());
        assert_eq!(result, 0);
        let result = leetcode_1143_opt("abcde".to_string(), "ace".to_string());
        assert_eq!(result, 3);
        let result = leetcode_1143_opt("abc".to_string(), "abc".to_string());
        assert_eq!(result, 3);
        let result = leetcode_1143_opt("abc".to_string(), "def".to_string());
        assert_eq!(result, 0);
    }


    #[test]
    fn test_leetcode_1964() {
        let result = leetcode_1964_iterative(vec![1, 2, 3, 2]);
        assert_eq!(result, vec![1, 2, 3, 3]);
        let result = leetcode_1964_iterative(vec![2, 2, 1]);
        assert_eq!(result, vec![1, 2, 1]);
        let result = leetcode_1964_iterative(vec![3, 1, 5, 6, 4, 2]);
        assert_eq!(result, vec![1, 1, 2, 3, 2, 2]);
    }

    #[test]
    fn test_leetcode_354() {
        let result = leetcode_354_iterative(vec![vec![5, 4], vec![6, 4], vec![6, 7], vec![2, 3]]);
        assert_eq!(result, 3);
        let result = leetcode_354_iterative(vec![vec![1, 1], vec![1, 1], vec![1, 1]]);
        assert_eq!(result, 1);
        let result = leetcode_354_iterative(create_vectors());
        assert_eq!(result, 100000);
    }

    #[test]
    fn test_leetcode_1027() {
        let result = leetcode_1027_iterative(vec![3, 6, 9, 12]);
        assert_eq!(result, 4);
        let result = leetcode_1027_iterative(vec![9, 4, 7, 2, 10]);
        assert_eq!(result, 3);
        let result = leetcode_1027_iterative(vec![20, 1, 15, 3, 10, 5, 8]);
        assert_eq!(result, 4);
        let result = leetcode_1027_iterative(vec![83, 20, 17, 43, 52, 78, 68, 45]);
        assert_eq!(result, 2);
    }

    #[test]
    fn test_leetcode_1218() {
        let result = leetcode_1218_iterative(vec![1, 2, 3, 4], 1);
        assert_eq!(result, 4);
        let result = leetcode_1218_iterative(vec![1, 3, 5, 7], 1);
        assert_eq!(result, 1);
        let result = leetcode_1218_iterative(vec![1, 5, 7, 8, 5, 3, 4, 2, 1], -2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_leetcode_646() {
        let result = leetcode_646_iterative(vec![vec![1, 2], vec![2, 3], vec![3, 4]]);
        assert_eq!(result, 2);
        let result = leetcode_646_iterative(vec![vec![1, 2], vec![7, 8], vec![4, 5]]);
        assert_eq!(result, 3);
        let result = leetcode_646_iterative(vec![
            vec![-10, -8],
            vec![8, 9],
            vec![-5, 0],
            vec![6, 10],
            vec![-6, -4],
            vec![1, 7],
            vec![9, 10],
            vec![-4, 7],
        ]);
        assert_eq!(result, 4);
    }
    #[test]
    fn test_leetcode_673() {
        let result = leetcode_673_iterative(vec![1, 3, 5, 4, 7]);
        assert_eq!(result, 2);
        let result = leetcode_673_iterative(vec![2, 2, 2, 2, 2]);
        assert_eq!(result, 5);
    }

    #[test]
    fn test_leetcode_300() {
        let result = leetcode_300_recursive(vec![10, 9, 2, 5, 3, 7, 101, 18]);
        assert_eq!(result, 4);
        let result = leetcode_300_recursive(vec![0, 1, 0, 3, 2, 3]);
        assert_eq!(result, 4);
        let result = leetcode_300_recursive(vec![7, 7, 7, 7, 7, 7, 7]);
        assert_eq!(result, 1);
        let result = leetcode_300_iterative(vec![10, 9, 2, 5, 3, 7, 101, 18]);
        assert_eq!(result, 4);
        let result = leetcode_300_iterative(vec![0, 1, 0, 3, 2, 3]);
        assert_eq!(result, 4);
        let result = leetcode_300_iterative(vec![7, 7, 7, 7, 7, 7, 7]);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_leetcode_115() {
        let result = leetcode_115("rabbbit", "rabbit");
        assert_eq!(result, 3);
        let result = leetcode_115("babgbag", "bag");
        assert_eq!(result, 5);
        let result = leetcode_115_dp("rabbbit", "rabbit");
        assert_eq!(result, 3);
        let result = leetcode_115_dp("babgbag", "bag");
        assert_eq!(result, 5);
        let result = leetcode_115_dp_opt("rabbbit", "rabbit");
        assert_eq!(result, 3);
        let result = leetcode_115_dp_opt("babgbag", "bag");
        assert_eq!(result, 5);
    }
    #[test]
    fn test_leetcode_712() {
        let result = leetcode_712("sea".to_string(), "eat".to_string());
        assert_eq!(result, 231);
        let result = leetcode_712("delete".to_string(), "leet".to_string());
        assert_eq!(result, 403);
        let result = leetcode_712_dp("sea".to_string(), "eat".to_string());
        assert_eq!(result, 231);
        let result = leetcode_712_dp("delete".to_string(), "leet".to_string());
        assert_eq!(result, 403);
        let result = leetcode_712_dp_opt("sea".to_string(), "eat".to_string());
        assert_eq!(result, 231);
        let result = leetcode_712_dp_opt("delete".to_string(), "leet".to_string());
        assert_eq!(result, 403);
    }

    #[test]
    fn test_leetcode_72() {
        let result = leetcode_72("horse".to_string(), "ros".to_string());
        assert_eq!(result, 3);
        let result = leetcode_72("intention".to_string(), "execution".to_string());
        assert!(result == 5);
        let result = leetcode_72_opt("horse".to_string(), "ros".to_string());
        assert_eq!(result, 3);
        let result = leetcode_72_opt("intention".to_string(), "execution".to_string());
        assert!(result == 5);
    }
    #[test]
    fn test_leetcode_516() {
        let result = leetcode_516_recursive("bbbab".to_string());
        assert!(result == 4);
        let result = leetcode_516_recursive("cbbd".to_string());
        assert!(result == 2);
        let result = leetcode_516_dp("bbbab".to_string());
        assert!(result == 4);
        let result = leetcode_516_dp("cbbd".to_string());
        assert!(result == 2);
    }

    #[test]
    fn test_leetcode_139() {
        let result = leetcode_139("leetcode".to_string(), vec_of_strings!["leet", "code"]);
        assert!(result);
        let result = leetcode_139("applepenapple".to_string(), vec_of_strings!["apple", "pen"]);
        assert!(result);
        let result = leetcode_139(
            "catsandog".to_string(),
            vec_of_strings!["cats", "dog", "sand", "and", "cat"],
        );
        assert!(!result);
        let result =
            leetcode_139_recursive("leetcode".to_string(), vec_of_strings!["leet", "code"]);
        assert!(result);
        let result =
            leetcode_139_recursive("applepenapple".to_string(), vec_of_strings!["apple", "pen"]);
        assert!(result);
        let result = leetcode_139_recursive(
            "catsandog".to_string(),
            vec_of_strings!["cats", "dog", "sand", "and", "cat"],
        );
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
        let grid = vec![
            vec!['1', '0', '1', '0', '0'],
            vec!['1', '0', '1', '1', '1'],
            vec!['1', '1', '1', '1', '1'],
            vec!['1', '0', '0', '1', '0'],
        ];

        let result = leetcode_221(&grid);
        assert_eq!(result, 4);
        let result = leetcode_221(&vec![vec!['0', '1'], vec!['1', '0']]);
        assert_eq!(result, 1);
        let result = leetcode_221(&vec![vec!['0']]);
        assert_eq!(result, 0);
        let result = leetcode_221_vector(&grid);
        assert_eq!(result, 4);
        let result = leetcode_221_vector(&vec![vec!['0', '1'], vec!['1', '0']]);
        assert_eq!(result, 1);
        let result = leetcode_221_vector(&vec![vec!['0']]);
        assert_eq!(result, 0);
        let result = leetcode_221_vector_iter(&grid);
        assert_eq!(result, 4);
        let result = leetcode_221_vector_iter(&vec![vec!['0', '1'], vec!['1', '0']]);
        assert_eq!(result, 1);
        let result = leetcode_221_vector_iter(&vec![vec!['0']]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_leetcode_931() {
        let result = leetcode_931(vec![vec![2, 1, 3], vec![6, 5, 4], vec![7, 8, 9]]);
        assert_eq!(result, 13);
        let result = leetcode_931(vec![vec![-19, 57], vec![-40, -5]]);
        assert_eq!(result, -59);
    }

    #[test]
    fn test_leetcode_63() {
        let big_grid = vec![
            vec![
                0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
                0, 1, 0, 0, 0, 1, 0, 0,
            ],
            vec![
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
                0, 1, 0, 0, 1, 0, 0, 0,
            ],
            vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 1, 0, 1,
            ],
            vec![
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0,
            ],
            vec![
                0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                0, 0, 1, 0, 0, 1, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 1, 0, 0,
            ],
            vec![
                1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                1, 0, 1, 1, 0, 0, 0, 1,
            ],
            vec![
                0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 0, 0, 0,
            ],
            vec![
                1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                0, 0, 1, 0, 0, 0, 1, 0,
            ],
            vec![
                0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                1, 0, 0, 1, 0, 0, 0, 1,
            ],
            vec![
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 1, 0, 0, 0,
            ],
            vec![
                1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
            ],
            vec![
                0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 1, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
            ],
            vec![
                0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 0, 1, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1,
                1, 1, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            vec![
                1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 1, 0,
            ],
            vec![
                0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 1, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 1, 1, 0, 0,
            ],
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ];
        let big_grid_result = 718991952;
        let result = leetcode_63(vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]]);
        assert_eq!(result, 2);
        let result = leetcode_63(vec![vec![0, 1], vec![0, 0]]);
        assert_eq!(result, 1);
        let result = leetcode_63(vec![vec![0, 0], vec![0, 1]]);
        assert_eq!(result, 0);
        let result = leetcode_63_memo(&vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]]);
        assert_eq!(result, 2);
        let result = leetcode_63_memo(&vec![vec![0, 1], vec![0, 0]]);
        assert_eq!(result, 1);
        let result = leetcode_63_memo(&vec![vec![0, 0], vec![0, 1]]);
        assert_eq!(result, 0);
        let result = leetcode_63_memo(&big_grid);
        assert_eq!(result, big_grid_result);
        let result = leetcode_63_iterative(&vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 0]]);
        assert_eq!(result, 2);
        let result = leetcode_63_iterative(&vec![vec![0, 1], vec![0, 0]]);
        assert_eq!(result, 1);
        let result = leetcode_63_iterative(&vec![vec![0, 0], vec![0, 1]]);
        assert_eq!(result, 0);
        let result = leetcode_63_iterative(&big_grid);
        assert_eq!(result, big_grid_result);
    }
    #[test]
    fn test_leetcode_120() {
        let result = leetcode_120(vec![vec![2], vec![3, 4], vec![6, 5, 7], vec![4, 1, 8, 3]]);
        assert_eq!(result, 11);
        let result = leetcode_120(vec![vec![-10]]);
        assert_eq!(result, -10);
    }
    #[test]
    fn test_leetcode_64() {
        let big_grid = vec![
            vec![3, 8, 6, 0, 5, 9, 9, 6, 3, 4, 0, 5, 7, 3, 9, 3],
            vec![0, 9, 2, 5, 5, 4, 9, 1, 4, 6, 9, 5, 6, 7, 3, 2],
            vec![8, 2, 2, 3, 3, 3, 1, 6, 9, 1, 1, 6, 6, 2, 1, 9],
            vec![1, 3, 6, 9, 9, 5, 0, 3, 4, 9, 1, 0, 9, 6, 2, 7],
            vec![8, 6, 2, 2, 1, 3, 0, 0, 7, 2, 7, 5, 4, 8, 4, 8],
            vec![4, 1, 9, 5, 8, 9, 9, 2, 0, 2, 5, 1, 8, 7, 0, 9],
            vec![6, 2, 1, 7, 8, 1, 8, 5, 5, 7, 0, 2, 5, 7, 2, 1],
            vec![8, 1, 7, 6, 2, 8, 1, 2, 2, 6, 4, 0, 5, 4, 1, 3],
            vec![9, 2, 1, 7, 6, 1, 4, 3, 8, 6, 5, 5, 3, 9, 7, 3],
            vec![0, 6, 0, 2, 4, 3, 7, 6, 1, 3, 8, 6, 9, 0, 0, 8],
            vec![4, 3, 7, 2, 4, 3, 6, 4, 0, 3, 9, 5, 3, 6, 9, 3],
            vec![2, 1, 8, 8, 4, 5, 6, 5, 8, 7, 3, 7, 7, 5, 8, 3],
            vec![0, 7, 6, 6, 1, 2, 0, 3, 5, 0, 8, 0, 8, 7, 4, 3],
            vec![0, 4, 3, 4, 9, 0, 1, 9, 7, 7, 8, 6, 4, 6, 9, 5],
            vec![6, 5, 1, 9, 9, 2, 2, 7, 4, 2, 7, 2, 2, 3, 7, 2],
            vec![7, 1, 9, 6, 1, 2, 7, 0, 9, 6, 6, 4, 4, 5, 1, 0],
            vec![3, 4, 9, 2, 8, 3, 1, 2, 6, 9, 7, 0, 2, 4, 2, 0],
            vec![5, 1, 8, 8, 4, 6, 8, 5, 2, 4, 1, 6, 2, 2, 9, 7],
        ];
        let big_grid_result = 83;
        let result = leetcode_64(vec![vec![1, 3, 1], vec![1, 5, 1], vec![4, 2, 1]]);
        assert_eq!(result, 7);
        let result = leetcode_64(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(result, 12);
        let result = leetcode_64_iterative(vec![vec![1, 3, 1], vec![1, 5, 1], vec![4, 2, 1]]);
        assert_eq!(result, 7);
        let result = leetcode_64_iterative(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        assert_eq!(result, 12);
        let result = leetcode_64_iterative(big_grid);
        assert_eq!(result, big_grid_result);
    }

    #[test]
    fn test_leetcode_62() {
        let result = leetcode_62(3, 7);
        assert_eq!(result, 28);
        let result = leetcode_62(3, 2);
        assert_eq!(result, 3);
        let result = leetcode_62_recursive(3, 7);
        assert_eq!(result, 28);
        let result = leetcode_62_recursive(3, 2);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_leetcode_740() {
        let result = leetcode_740(vec![3, 4, 2]);
        assert_eq!(result, 6);
        let result = leetcode_740(vec![2, 2, 3, 3, 3, 4]);
        assert_eq!(result, 9);
        let result = leetcode_740_iterative_cpp(vec![3, 4, 2]);
        assert_eq!(result, 6);
        let result = leetcode_740_iterative_cpp(vec![2, 2, 3, 3, 3, 4]);
        assert_eq!(result, 9);
        let result = leetcode_740_iterative_rust(vec![3, 4, 2]);
        assert_eq!(result, 6);
        let result = leetcode_740_iterative_rust(vec![2, 2, 3, 3, 3, 4]);
        assert_eq!(result, 9);
    }

    #[test]
    fn test_leetcode_198() {
        let long_vec = vec![
            114, 117, 207, 117, 235, 82, 90, 67, 143, 146, 53, 108, 200, 91, 80, 223, 58, 170, 110,
            236, 81, 90, 222, 160, 165, 195, 187, 199, 114, 235, 197, 187, 69, 129, 64, 214, 228,
            78, 188, 67, 205, 94, 205, 169, 241, 202, 144, 240,
        ];
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
