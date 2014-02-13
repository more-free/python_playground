__author__ = 'morefree'

# interview problems from leetcode, codeEval, codeforces


# leetcode, Longest Substring Without Repeating Characters
class Solution:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        if s == '': return 0
        low, maxLen = 0, 0
        used = {}
        for high in range(len(s)):
            key = s[high]
            if key not in used:
                used[key] = 1
            else:
                maxLen = max(maxLen, high-low)
                while s[low] != key:
                    used.__delitem__(s[low])
                    low += 1
                low += 1
        maxLen = max(maxLen, high-low+1)
        return maxLen

# leetcode, Sudoku solver
class Solution:
    # @param board, a 9x9 2D array
    # Solve the Sudoku by modifying the input board in-place.
    # Do not return any value.
    def __init__(self):
        self.LENGTH = 9
        self.row_table = self.two_dim_list(self.LENGTH, 0)
        self.col_table = self.two_dim_list(self.LENGTH, 0)
        self.block_table = self.two_dim_list(self.LENGTH, 0)

    def two_dim_list(self, length, val):
        return [[val]*length for i in range(length)]

    def solveSudoku(self, board):
        self.init_used_table(board)
        self.fill(board, 0, 0)

    def init_used_table(self, board):
        for i in range(self.LENGTH):
            for j in range(self.LENGTH):
                self.row_table[i][j] = self.col_table[i][j] = self.block_table[i][j] = 0

        for i in range(self.LENGTH):
            for j in range(self.LENGTH):
                if board[i][j] != '.':
                    self.set_ds(i, j, int(board[i][j]) - 1)

    def set_ds(self, row, col, val):
        self.row_table[row][val] = 1
        self.col_table[col][val] = 1
        self.block_table[row/3*3+col/3][val] = 1

    def restore_ds(self, row, col, val):
        self.row_table[row][val] = 0
        self.col_table[col][val] = 0
        self.block_table[row/3*3+col/3][val] = 0

    def in_ds(self, row, col, val):
        return self.row_table[row][val] == 1 or \
            self.col_table[col][val] == 1 or \
            self.block_table[row/3*3+col/3][val] == 1

    def update_idx(self, row, col):
        if col == self.LENGTH - 1:
            return row + 1, 0
        else:
            return row, col + 1

    def fill(self, board, row, col):
        if row == self.LENGTH:
            return True

        if board[row][col] != '.':
            next_row, next_col = self.update_idx(row, col)
            return self.fill(board, next_row, next_col)

        for val in range(self.LENGTH):
            if not self.in_ds(row, col, val):
                board[row][col] = str(val+1)
                self.set_ds(row, col, val)
                next_row, next_col = self.update_idx(row, col)
                if self.fill(board, next_row, next_col):
                    return True
                self.restore_ds(row, col, val)
                board[row][col] = '.'

        return False

# next permutation
# 1,2,3  1,3,2
# 3,2,1  1,2,3
# 1,1,5  1,5,1
class Solution:
    # @param num, a list of integer
    # @return a list of integer
    def nextPermutation(self, num):
        for i in reversed(range(1, len(num))):
            if num[i] > num[i-1]:
                self.rerange(num, i-1)
                return num
        num[:] = reversed(num)
        return num

    def rerange(self, num, start):
        for i in reversed(range(start+1, len(num))):
            if num[i] > num[start]:
                num[start], num[i] = num[i], num[start]
                num[start+1:] = reversed(num[start+1:])
                break


# leetcode regular expression match
# note : memorized search
class Solution:
    def __init__(self):
        self.sLen = 0
        self.pLen = 0

    # @return a boolean
    def isMatch(self, s, p):
        self.sLen = len(s)
        self.pLen = len(p)
        match = [[None]*self.pLen for i in range(self.sLen + 1)]
        return self.isMatchHelper(s, p, 0, 0, match)

    def isMatchHelper(self, s, p, si, pi, match):
        if pi == self.pLen:
            return si == self.sLen
        if si == self.sLen:
            match[si][pi] = (pi < self.pLen-1) and (p[pi+1] == '*') and self.isMatchHelper(s, p, si, pi+2, match)
            return match[si][pi]
        if si < self.sLen and match[si][pi] != None:
            return match[si][pi]

        if pi < self.pLen - 1 and p[pi+1] == '*':
            match[si][pi] = (self.key_match(s[si], p[pi]) and \
                             self.isMatchHelper(s, p, si+1, pi, match)) or \
                             self.isMatchHelper(s, p, si, pi+2, match)
        else:
            match[si][pi] = self.key_match(s[si], p[pi]) and \
                            self.isMatchHelper(s, p, si+1, pi+1, match)
        return match[si][pi]

    def key_match(self, skey, pkey):
        return skey == pkey or '.' == pkey


# 3 sum
class Solution:
    # @return a list of lists of length 3, [[val1,val2,val3]]
    def threeSum(self, num):
        length = len(num)
        ans = []
        num.sort()
        for i in range(length-2):
            if i > 0 and num[i-1] == num[i]:
                continue
            low, high = i+1, length-1
            while low < high:
                if low > i + 1 and num[low-1] == num[low]:
                    low += 1
                    continue
                if high < length - 1 and num[high] == num[high+1]:
                    high -= 1
                    continue
                sum = num[low] + num[high]
                if sum == -num[i]:
                    ans.append([num[i], num[low], num[high]])
                    low, high = low + 1, high - 1
                elif sum < -num[i]:
                    low += 1
                else:
                    high -= 1
        return ans


# implement strStr()
class Solution:
    # @param haystack, a string
    # @param needle, a string
    # @return a string or None
    def strStr(self, haystack, needle):
        len_haystack, len_needle = len(haystack), len(needle)
        if len_haystack < len_needle:
            return None
        if len_needle == 0 and len_haystack == 0:
            return ""

        hash_code, needle_hash_code = 0, 0
        for i in range(len_needle):
            hash_code += ord(haystack[i])
            needle_hash_code += ord(needle[i])


        for i in range(len_haystack - len_needle + 1):
            if hash_code == needle_hash_code:
                if needle == haystack[i:i+len_needle]:
                    return haystack[i:]
            hash_code -= ord(haystack[i])
            if i + len_needle < len_haystack:
                hash_code += ord(haystack[i+len_needle])

        return None


class Solution:
    def countAndSay(self, n):
        ans = '1'
        for i in range(1, n):
            cnt, local = 1, []
            for j in range(1, len(ans)):
                if ans[j-1] == ans[j]:
                    cnt += 1
                else:
                    local.append(str(cnt))
                    local.append(ans[j-1])
                    cnt = 1
            local.append(str(cnt))
            local.append(ans[-1])
            ans = ''.join(local)

        return ans

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# a variation of leetcode "add two numbers" : 4->1->2 stands for 412 instead of 214
class Solution:
    # @return a ListNode
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(-1)
        carry = self.addTwoLists(dummy, l1, l2, self.getlen(l1), self.getlen(l2))
        if carry != 0:
            self.insert(dummy, carry)
        return dummy.next

    def getlen(self, list):
        length = 0
        while list != None:
            length += 1
            list = list.next
        return length

    def addTwoLists(self, dummy, l1, l2, len1, len2):
        carry, sum = 0, 0
        if len1 > len2:
            carry = self.addTwoLists(dummy, l1.next, l2, len1-1, len2)
            sum = l1.val + carry
        elif len2 > len1:
            carry = self.addTwoLists(dummy, l1, l2.next, len1, len2-1)
            sum = l2.val + carry
        else:
            if len1 == 0:
                return 0
            else:
                carry = self.addTwoLists(dummy, l1.next, l2.next, len1-1, len2-1)
                sum = l1.val + l2.val + carry

        carry = sum / 10
        self.insert(dummy, sum % 10)
        return carry

    def insert(self, node, val):
        next = node.next
        node.next = ListNode(val)
        node.next.next = next


class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(-1)
        self.addTwoLists(l1, l2, 0, dummy)
        return dummy.next

    def addTwoLists(self, l1, l2, carry, dummy):
        if l1 == l2 == None:
            if carry != 0:
                dummy.next = ListNode(carry)
            return

        sum = self.get_val(l1) + self.get_val(l2) + carry
        dummy.next = ListNode(sum % 10)
        self.addTwoLists(self.get_next(l1), self.get_next(l2), sum / 10, dummy.next)

    def get_val(self, list):
        return 0 if list == None else list.val

    def get_next(self, list):
        return None if list == None else list.next


# Triangle
class Solution:
    # @param triangle, a list of lists of integers
    # @return an integer
    def minimumTotal(self, triangle):
        levels = len(triangle)
        size = len(triangle[levels - 1])

        maxpath = triangle[levels-1][:]
        for level in reversed(range(levels-1)):
            size -= 1
            for i in range(size):
                maxpath[i] = triangle[level][i] + min(maxpath[i], maxpath[i+1])
        return maxpath[0]


# Wildcard matching
# note : using memorized search, causing Memory Limit Exceeded on leetcode
# improve : change to dynamic programming with rolling array (later on this)
class Solution:
    # @param s, an input string
    # @param p, a pattern string
    # @return a boolean
    def __init__(self):
        self.slen = 0
        self.plen = 0

    def isMatch(self, s, p):
        self.slen = len(s)
        self.plen = len(p)
        dp = [[None]*self.plen for i in range(self.slen + 1)]
        return self.match(dp, s, p, 0, 0)

    def match(self, dp, s, p, si, pi):
        if pi == self.plen:
            return si == self.slen
        if si == self.slen:
            dp[si][pi] = True
            for i in range(pi, self.plen):
                if p[pi] != '*':
                    dp[si][pi] = False
                    break
            return dp[si][pi]

        if dp[si][pi] != None:
            return dp[si][pi]

        if s[si] == p[pi] or p[pi] == '?':
            dp[si][pi] = self.match(dp, s, p, si+1, pi+1)
        elif p[pi] == '*':
            dp[si][pi] = self.match(dp, s, p, si, pi+1) or \
                self.match(dp, s, p, si+1, pi)
        else:
            dp[si][pi] = False

        return dp[si][pi]


# leetcode, anagrams
class Solution:
    # @param strs, a list of strings
    # @return a list of strings
    def anagrams(self, strs):
        hash = {}
        for str in strs:
            key = ''.join(sorted(str))
            if key in hash:
                hash[key].append(str)
            else:
                hash[key] = [str]

        ans = []
        for key, val in hash.items():
            if len(val) > 1:
                for e in val:
                    ans.append(e)
        return ans


# maximum subarray
class Solution:
    # @param A, a list of integers
    # @return an integer
    def maxSubArray(self, A):
        length = len(A)
        if length == 0:
            return 0
        maxsum = A[0]
        localsum = A[0]
        for i in range(1, length):
            localsum = max(localsum + A[i], A[i])
            maxsum = max(maxsum, localsum)
        return maxsum


# combination sum II
#  note : on leetcode it causes Compile Error. Why ?
class Solution:
    # @param candidates, a list of integers
    # @param target, integer
    # @return a list of lists of integers
    def combinationSum2(self, candidates, target):
        candidates.sort()
        ans = []
        self.dfs(candidates, 0, 0, target, [], ans)
        return ans

    def dfs(self, candidates, s, local_sum, target, local_ans, global_ans):
        if local_sum == target:
            global_ans.append(local_ans[:])
        elif local_sum > target or s == len(candidates):
            return
        else:
            val = candidates[s]
            local_ans.append(val)
            self.dfs(candidates, s + 1, local_sum + val, target, local_ans, global_ans)
            local_ans.pop()
            while s < len(candidates) and candidates[s] == val:
                s += 1
            self.dfs(candidates, s, local_sum, target, local_ans, global_ans)


# climbing stairs
class Solution:
    # @param n, an integer
    # @return an integer
    def climbStairs(self, n):
        if n <= 0:
            return 0
        if n == 1 or n == 2:
            return n
        prev2, prev1, cur = 1, 2, 0
        for i in range(3, n+1):
            cur = prev2 + prev1
            prev2 = prev1
            prev1 = cur
        return cur


# Minimum Window Substring
class Solution:
    # @return a string
    def minWindow(self, S, T):
        target = {}
        for t in T:
            target[t] = target.setdefault(t, 0) + 1

        ans = ""
        found = {}
        low, high, fullfilled = 0, 0, 0
        for high in range(len(S)):
            if S[high] in target:
                found[S[high]] = found.setdefault(S[high], 0) + 1
                if found[S[high]] == target[S[high]]:
                    fullfilled += 1
                if fullfilled == len(target):
                        # moving 'low' to shrink the window
                    while low <= high:
                        if S[low] in target:
                            if found[S[low]] > target[S[low]]:
                                found[S[low]] -= 1
                            else:
                                local_ans = S[low : high+1]
                                ans = local_ans if ans == "" or len(local_ans) < len(ans) else ans
                                break
                        low += 1
        return ans


# leetcode maximal rectangle
class Solution:
    # @param matrix, a list of lists of 1 length string
    # @return an integer
    def maximalRectangle(self, matrix):
        if len(matrix) == 0:
            return 0
        rows, cols = len(matrix), len(matrix[0])
        height = [0 for i in range(cols)]
        left_border = [0 for i in range(cols)]
        right_border = [cols-1 for i in range(cols)]
        max_area = 0

        for row in range(rows):
            leftmost_nonzero, rightmost_nonzero = 0, cols - 1
            for col in range(cols):
                if matrix[row][col] == '1':
                    height[col] += 1
                    left_border[col] = max(left_border[col], leftmost_nonzero)
                else:
                    leftmost_nonzero = col + 1
                    height[col] = 0
                    left_border[col] = 0

            for col in reversed(range(cols)):
                if matrix[row][col] == '1':
                    right_border[col] = min(right_border[col], rightmost_nonzero)
                    max_area = max(max_area, (right_border[col] - left_border[col] + 1) * height[col])
                else:
                    rightmost_nonzero = col - 1
                    right_border[col] = cols - 1

        return max_area


# merge K sorted list
# note : cause Compile Error on Leetcode.  (why ?)
import heapq

class HeapKey:
    def __init__(self, node):
        self.node = node
    def __cmp__(self, other):
        return cmp(self.node.val, other.node.val)

class Solution:
    # @param a list of ListNode
    # @return a ListNode
    def mergeKLists(self, lists):
        heads = [HeapKey(lists[i]) for i in range(len(lists)) if lists[i]]
        heapq.heapify(heads)

        dummy = ListNode(-1)
        cur = dummy
        while len(heads) > 0:
            nextItem = heapq.heappop(heads)
            cur.next = nextItem.node
            cur = cur.next
            if cur.next:
                heapq.heappush(heads, HeapKey(cur.next))
        return dummy.next


# leetcode, Substring with Concatenation of All Words
# S: "barfoothefoobarman"
# L: ["foo", "bar"]
# You should return the indices: [0,9].
# note : use brute force + Robin-Karp, is there a better solution ?
class Solution:
    # @param S, a string
    # @param L, a list of string
    # @return a list of integer
    def findSubstring(self, S, L):
        target = {}
        for word in L:
            target[word] = target.setdefault(word, 0) + 1
        target_hash = 0
        for li in L:
            target_hash += sum(ord(letter) for letter in li)
        prev_hash = [0]
        return [i for i in range(len(S)) if self.matches(S, i, L, target, len(L[0]), target_hash, prev_hash)]

    def matches(self, S, cur, L, target, length, target_hash, prev_hash):
        if len(S) - cur < len(L) * length:
            return False
        if prev_hash[0] != 0:
            prev_hash[0] = prev_hash[0] - ord(S[cur-1]) + ord(S[cur+len(L)*length-1])
        else:
            prev_hash[0] = sum([ord(S[i]) for i in range(cur, cur+len(L)*length)])

        if prev_hash[0] != target_hash:
            return False

        found = {}
        for i in range(len(L)):
            word = S[cur+i*length:cur+(i+1)*length]
            num = found[word] = found.setdefault(word, 0) + 1
            if word not in target or target[word] < num:
                return False

        return True


# Search Insert Position
class Solution:
    # @param A, a list of integers
    # @param target, an integer to be inserted
    # @return integer
    def searchInsert(self, A, target):
        low, high = 0, len(A) - 1
        while low + 1 < high:
            mid = (low + high) / 2
            if A[mid] == target:
                return mid
            elif A[mid] < target:
                low = mid
            else:
                high = mid
        if A[low] == target:
            return low
        if A[high] == target:
            return high
        if A[low] < target < A[high]:
            return high
        if A[low] > target:
            return low
        if A[high] < target:
            return high + 1


# first missing positive
class Solution:
    # @param A, a list of integers
    # @return an integer
    def firstMissingPositive(self, A):
        pos = -1
        for i in range(len(A)):
            if A[i] > 0:
                pos += 1
                A[pos], A[i] = A[i], A[pos]
        if pos == -1:
            return 1

        for i in range(pos+1):
            val = abs(A[i])
            if 1 <= val <= pos + 1:
                if A[val - 1] < 0:
                    continue
                A[val - 1] = -A[val - 1]
        for i in range(pos + 1):
            if A[i] > 0:
                return i + 1
        return pos + 2


# permutations
class Solution:
    # @param num, a list of integer
    # @return a list of lists of integers
    def permute(self, num):
        ans = []
        self.permutate(num, 0, [], ans)
        return ans

    def permutate(self, num, cur, local_ans, ans):
        if cur == len(num):
            ans.append(local_ans[:])
        else:
            for i in range(cur, len(num)):
                num[cur], num[i] = num[i], num[cur]
                local_ans.append(num[cur])
                self.permutate(num, cur+1, local_ans, ans)
                local_ans.pop()
                num[cur], num[i] = num[i], num[cur]


# permutations II
class Solution:
    # @param num, a list of integer
    # @return a list of lists of integers
    def permuteUnique(self, num):
        ans = []
        self.permutate(num, 0, [], ans)
        return ans

    def permutate(self, num, cur, local_ans, ans):
        if cur == len(num):
            ans.append(local_ans[:])
        else:
            used = {}
            for i in range(cur, len(num)):
                if num[i] not in used:
                    used[num[i]] = True
                    num[i], num[cur] = num[cur], num[i]
                    local_ans.append(num[cur])
                    self.permutate(num, cur + 1, local_ans, ans)
                    local_ans.pop()
                    num[i], num[cur] = num[cur], num[i]
            used.clear()

s = Solution()
t = s.permuteUnique([-1, 0, 0, 1, 2])
print len(t), t

# merge intervals
class Interval:
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

class Solution:
    # @param intervals, a list of Interval
    # @return a list of Interval
    def merge(self, intervals):
        intervals.sort(key=lambda x: x.start)
        ans = []
        for i in range(len(intervals)):
            if len(ans) == 0:
                ans.append(intervals[i])
            else:
                if intervals[i].start > ans[-1].end:
                    ans.append(intervals[i])
                else:
                    ans[-1].end = max(intervals[i].end, ans[-1].end)
        return ans







