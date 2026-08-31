"""
Hardcoded DSA question bank for the AI Interview Agent.

Questions are grouped by difficulty and are fully self-contained — every constraint,
example, and expected output is included so the candidate has everything they need
without any verbal explanation from the interviewer.

The agent picks ONE question from the appropriate tier based on the candidate's
resume and answers, then passes the FULL text verbatim to show_coding_question().
"""

from typing import TypedDict


class DSAQuestion(TypedDict):
    id: str
    title: str
    difficulty: str  # "easy" | "medium" | "hard"
    tags: list[str]
    body: str  # Complete problem statement shown in the code editor


# ---------------------------------------------------------------------------
# EASY — Suitable for freshers / interns / junior candidates
# ---------------------------------------------------------------------------
EASY: list[DSAQuestion] = [
    {
        "id": "E1",
        "title": "Two Sum",
        "difficulty": "easy",
        "tags": ["array", "hash-map"],
        "body": (
            "Problem: Two Sum\n"
            "Difficulty: Easy\n"
            "─────────────────────────────────────────\n\n"
            "Given an array of integers `nums` and a target integer `target`,\n"
            "return the indices of the two numbers that add up to `target`.\n\n"
            "You may assume that each input has exactly one solution,\n"
            "and you may not use the same element twice.\n"
            "Return the answer in any order.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : nums = [2, 7, 11, 15], target = 9\n"
            "  Output: [0, 1]\n"
            "  Reason: nums[0] + nums[1] = 2 + 7 = 9\n\n"
            "  Input : nums = [3, 2, 4], target = 6\n"
            "  Output: [1, 2]\n\n"
            "  Input : nums = [3, 3], target = 6\n"
            "  Output: [0, 1]\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 2 ≤ nums.length ≤ 10⁴\n"
            "  • -10⁹ ≤ nums[i] ≤ 10⁹\n"
            "  • -10⁹ ≤ target ≤ 10⁹\n"
            "  • Exactly one valid answer exists.\n\n"
            "Bonus: Can you solve it in O(n) time?\n"
        ),
    },
    {
        "id": "E2",
        "title": "Valid Parentheses",
        "difficulty": "easy",
        "tags": ["stack", "string"],
        "body": (
            "Problem: Valid Parentheses\n"
            "Difficulty: Easy\n"
            "─────────────────────────────────────────\n\n"
            "Given a string `s` containing only the characters\n"
            "'(', ')', '{', '}', '[' and ']',\n"
            "determine if the input string is valid.\n\n"
            "A string is valid if:\n"
            "  1. Every open bracket is closed by the same type of bracket.\n"
            "  2. Open brackets are closed in the correct order.\n"
            "  3. Every close bracket has a corresponding open bracket.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : s = \"()\"\n"
            "  Output: True\n\n"
            "  Input : s = \"()[]{}\"\n"
            "  Output: True\n\n"
            "  Input : s = \"(]\"\n"
            "  Output: False\n\n"
            "  Input : s = \"([)]\"\n"
            "  Output: False\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 1 ≤ s.length ≤ 10⁴\n"
            "  • s consists of only '(', ')', '{', '}', '[', ']'\n"
        ),
    },
    {
        "id": "E3",
        "title": "Reverse a Linked List",
        "difficulty": "easy",
        "tags": ["linked-list", "pointers"],
        "body": (
            "Problem: Reverse a Linked List\n"
            "Difficulty: Easy\n"
            "─────────────────────────────────────────\n\n"
            "Given the head of a singly linked list, reverse the list\n"
            "and return the reversed list's head.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : 1 -> 2 -> 3 -> 4 -> 5\n"
            "  Output: 5 -> 4 -> 3 -> 2 -> 1\n\n"
            "  Input : 1 -> 2\n"
            "  Output: 2 -> 1\n\n"
            "─────────────────────────────────────────\n"
            "Node definition (already provided):\n\n"
            "  class ListNode:\n"
            "      def __init__(self, val=0, next=None):\n"
            "          self.val = val\n"
            "          self.next = next\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 0 <= number of nodes <= 5000\n"
            "  • -5000 <= Node.val <= 5000\n\n"
            "Bonus: Can you do it iteratively AND recursively?\n"
        ),
    },
    {
        "id": "E4",
        "title": "Find the Duplicate Number",
        "difficulty": "easy",
        "tags": ["array", "sorting", "hash-set"],
        "body": (
            "Problem: Find the Duplicate Number\n"
            "Difficulty: Easy\n"
            "─────────────────────────────────────────\n\n"
            "Given an array `nums` of n+1 integers where each integer\n"
            "is in the range [1, n] inclusive, there is exactly one\n"
            "repeated number. Find and return that duplicate.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : nums = [1, 3, 4, 2, 2]\n"
            "  Output: 2\n\n"
            "  Input : nums = [3, 1, 3, 4, 2]\n"
            "  Output: 3\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 1 <= n <= 10^5\n"
            "  • nums.length == n + 1\n"
            "  • 1 <= nums[i] <= n\n"
            "  • Exactly one value is duplicated (may appear more than twice)\n\n"
            "Bonus: Can you solve it without modifying the array and using O(1) extra space?\n"
        ),
    },
]

# ---------------------------------------------------------------------------
# MEDIUM — Suitable for mid-level / 1-3 years experience
# ---------------------------------------------------------------------------
MEDIUM: list[DSAQuestion] = [
    {
        "id": "M1",
        "title": "Longest Substring Without Repeating Characters",
        "difficulty": "medium",
        "tags": ["sliding-window", "hash-map", "string"],
        "body": (
            "Problem: Longest Substring Without Repeating Characters\n"
            "Difficulty: Medium\n"
            "─────────────────────────────────────────\n\n"
            "Given a string `s`, find the length of the longest substring\n"
            "that contains no repeating characters.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : s = \"abcabcbb\"\n"
            "  Output: 3\n"
            "  Reason: \"abc\" has length 3\n\n"
            "  Input : s = \"bbbbb\"\n"
            "  Output: 1\n\n"
            "  Input : s = \"pwwkew\"\n"
            "  Output: 3\n"
            "  Reason: \"wke\" has length 3\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 0 <= s.length <= 5 x 10^4\n"
            "  • s consists of English letters, digits, symbols, and spaces\n\n"
            "Hint: Think sliding window + a hash map tracking last seen index.\n"
        ),
    },
    {
        "id": "M2",
        "title": "Product of Array Except Self",
        "difficulty": "medium",
        "tags": ["array", "prefix-product"],
        "body": (
            "Problem: Product of Array Except Self\n"
            "Difficulty: Medium\n"
            "─────────────────────────────────────────\n\n"
            "Given an integer array `nums`, return an array `answer` where\n"
            "`answer[i]` equals the product of all elements of `nums` EXCEPT\n"
            "`nums[i]`.\n\n"
            "You must solve it in O(n) time and WITHOUT using the division operator.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : nums = [1, 2, 3, 4]\n"
            "  Output: [24, 12, 8, 6]\n\n"
            "  Input : nums = [-1, 1, 0, -3, 3]\n"
            "  Output: [0, 0, 9, 0, 0]\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 2 <= nums.length <= 10^5\n"
            "  • -30 <= nums[i] <= 30\n"
            "  • The product of any prefix or suffix fits in a 32-bit integer.\n\n"
            "Bonus: Can you do it with O(1) extra space (output array excluded)?\n"
        ),
    },
    {
        "id": "M3",
        "title": "Validate Binary Search Tree",
        "difficulty": "medium",
        "tags": ["tree", "BST", "recursion", "DFS"],
        "body": (
            "Problem: Validate Binary Search Tree\n"
            "Difficulty: Medium\n"
            "─────────────────────────────────────────\n\n"
            "Given the root of a binary tree, determine if it is a valid\n"
            "Binary Search Tree (BST).\n\n"
            "A valid BST satisfies:\n"
            "  • The left subtree contains only nodes with keys LESS THAN the node's key.\n"
            "  • The right subtree contains only nodes with keys GREATER THAN the node's key.\n"
            "  • Both left and right subtrees must also be valid BSTs.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input tree:\n"
            "        2\n"
            "       / \\\n"
            "      1   3\n"
            "  Output: True\n\n"
            "  Input tree:\n"
            "        5\n"
            "       / \\\n"
            "      1   4\n"
            "         / \\\n"
            "        3   6\n"
            "  Output: False\n"
            "  Reason: Node 4 is in right subtree of 5 but 4 < 5.\n\n"
            "─────────────────────────────────────────\n"
            "Node definition (already provided):\n\n"
            "  class TreeNode:\n"
            "      def __init__(self, val=0, left=None, right=None):\n"
            "          self.val = val\n"
            "          self.left = left\n"
            "          self.right = right\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 1 <= number of nodes <= 10^4\n"
            "  • -2^31 <= Node.val <= 2^31 - 1\n"
        ),
    },
    {
        "id": "M4",
        "title": "Maximum Subarray (Kadane's Algorithm)",
        "difficulty": "medium",
        "tags": ["array", "dynamic-programming", "kadane"],
        "body": (
            "Problem: Maximum Subarray\n"
            "Difficulty: Medium\n"
            "─────────────────────────────────────────\n\n"
            "Given an integer array `nums`, find the contiguous subarray\n"
            "(containing at least one number) which has the largest sum,\n"
            "and return that sum.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]\n"
            "  Output: 6\n"
            "  Reason: [4, -1, 2, 1] has the largest sum = 6\n\n"
            "  Input : nums = [1]\n"
            "  Output: 1\n\n"
            "  Input : nums = [5, 4, -1, 7, 8]\n"
            "  Output: 23\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 1 <= nums.length <= 10^5\n"
            "  • -10^4 <= nums[i] <= 10^4\n\n"
            "Bonus: If you find an O(n) solution, can you also describe\n"
            "       a divide-and-conquer approach (O(n log n))?\n"
        ),
    },
    {
        "id": "M5",
        "title": "LRU Cache",
        "difficulty": "medium",
        "tags": ["design", "hash-map", "doubly-linked-list"],
        "body": (
            "Problem: LRU Cache\n"
            "Difficulty: Medium\n"
            "─────────────────────────────────────────\n\n"
            "Design a data structure that follows the Least Recently Used (LRU)\n"
            "cache eviction policy.\n\n"
            "Implement the LRUCache class:\n"
            "  • LRUCache(capacity)  — Initialize with a positive capacity.\n"
            "  • get(key)            — Return the value if key exists, else -1.\n"
            "                         Mark this key as most recently used.\n"
            "  • put(key, value)     — Insert or update the key-value pair.\n"
            "                         If cache exceeds capacity, evict the\n"
            "                         least recently used item first.\n\n"
            "Both get and put must run in O(1) average time.\n\n"
            "─────────────────────────────────────────\n"
            "Example:\n\n"
            "  cache = LRUCache(2)\n"
            "  cache.put(1, 1)   # cache: {1:1}\n"
            "  cache.put(2, 2)   # cache: {1:1, 2:2}\n"
            "  cache.get(1)      # returns 1, cache: {2:2, 1:1}\n"
            "  cache.put(3, 3)   # evicts key 2, cache: {1:1, 3:3}\n"
            "  cache.get(2)      # returns -1 (not found)\n"
            "  cache.put(4, 4)   # evicts key 1, cache: {3:3, 4:4}\n"
            "  cache.get(1)      # returns -1\n"
            "  cache.get(3)      # returns 3\n"
            "  cache.get(4)      # returns 4\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 1 <= capacity <= 3000\n"
            "  • 0 <= key <= 10^4\n"
            "  • 0 <= value <= 10^5\n"
            "  • At most 2 x 10^5 calls to get and put\n"
        ),
    },
]

# ---------------------------------------------------------------------------
# HARD — Suitable for senior / 3+ years / strong candidates
# ---------------------------------------------------------------------------
HARD: list[DSAQuestion] = [
    {
        "id": "H1",
        "title": "Trapping Rain Water",
        "difficulty": "hard",
        "tags": ["array", "two-pointers", "stack"],
        "body": (
            "Problem: Trapping Rain Water\n"
            "Difficulty: Hard\n"
            "─────────────────────────────────────────\n\n"
            "Given n non-negative integers representing an elevation map where\n"
            "the width of each bar is 1, compute how much water it can trap\n"
            "after raining.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : height = [0,1,0,2,1,0,1,3,2,1,2,1]\n"
            "  Output: 6\n\n"
            "  Input : height = [4,2,0,3,2,5]\n"
            "  Output: 9\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • n == height.length\n"
            "  • 1 <= n <= 2 x 10^4\n"
            "  • 0 <= height[i] <= 10^5\n\n"
            "Bonus: Can you solve it in O(n) time and O(1) space?\n"
        ),
    },
    {
        "id": "H2",
        "title": "Word Ladder (BFS Shortest Path)",
        "difficulty": "hard",
        "tags": ["BFS", "graph", "string", "hash-set"],
        "body": (
            "Problem: Word Ladder\n"
            "Difficulty: Hard\n"
            "─────────────────────────────────────────\n\n"
            "Given a beginWord, an endWord, and a wordList, find the length\n"
            "of the shortest transformation sequence from beginWord to endWord\n"
            "such that:\n"
            "  1. Only one letter can be changed at a time.\n"
            "  2. Each transformed word must exist in wordList.\n\n"
            "Return the number of words in the shortest sequence, or 0 if\n"
            "no such sequence exists.\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : beginWord = \"hit\"\n"
            "          endWord   = \"cog\"\n"
            "          wordList  = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\",\"cog\"]\n"
            "  Output: 5\n"
            "  Path  : hit -> hot -> dot -> dog -> cog\n\n"
            "  Input : beginWord = \"hit\"\n"
            "          endWord   = \"cog\"\n"
            "          wordList  = [\"hot\",\"dot\",\"dog\",\"lot\",\"log\"]\n"
            "  Output: 0\n"
            "  Reason: 'cog' is not in wordList, no valid path.\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • 1 <= beginWord.length <= 10\n"
            "  • endWord.length == beginWord.length\n"
            "  • 1 <= wordList.length <= 5000\n"
            "  • All words are the same length, lowercase English letters only.\n"
            "  • beginWord != endWord\n"
        ),
    },
    {
        "id": "H3",
        "title": "Median of Two Sorted Arrays",
        "difficulty": "hard",
        "tags": ["binary-search", "array", "divide-and-conquer"],
        "body": (
            "Problem: Median of Two Sorted Arrays\n"
            "Difficulty: Hard\n"
            "─────────────────────────────────────────\n\n"
            "Given two sorted arrays `nums1` and `nums2` of size m and n\n"
            "respectively, return the median of the two sorted arrays.\n\n"
            "The overall runtime complexity must be O(log(m + n)).\n\n"
            "─────────────────────────────────────────\n"
            "Examples:\n\n"
            "  Input : nums1 = [1, 3],  nums2 = [2]\n"
            "  Output: 2.0\n"
            "  Merged: [1, 2, 3] -> median is 2.0\n\n"
            "  Input : nums1 = [1, 2],  nums2 = [3, 4]\n"
            "  Output: 2.5\n"
            "  Merged: [1, 2, 3, 4] -> median is (2 + 3) / 2 = 2.5\n\n"
            "─────────────────────────────────────────\n"
            "Constraints:\n"
            "  • nums1.length == m,  nums2.length == n\n"
            "  • 0 <= m, n <= 1000\n"
            "  • 1 <= m + n <= 2000\n"
            "  • -10^6 <= nums1[i], nums2[i] <= 10^6\n\n"
            "Note: A brute-force O(m+n) merge + median is expected —\n"
            "      talk through WHY the binary search approach is better.\n"
        ),
    },
]

# ---------------------------------------------------------------------------
# Flat lookup map: id -> question (for quick access in tools.py / agent.py)
# ---------------------------------------------------------------------------
ALL_QUESTIONS: dict[str, DSAQuestion] = {
    q["id"]: q for q in EASY + MEDIUM + HARD
}


def get_question_bank_prompt() -> str:
    """
    Returns a formatted string listing all available questions by tier.
    Injected into the system prompt so the LLM can pick one by ID and
    pass its full body text verbatim to show_coding_question().
    """
    lines = [
        "\n\n─────────────────────────────────────────",
        "CODING QUESTION BANK",
        "─────────────────────────────────────────",
        "When it is time for the coding exercise, pick ONE question from the bank below",
        "that best matches the candidate's experience level.",
        "Call show_coding_question() and pass the COMPLETE 'body' text VERBATIM as question_text.",
        "Do NOT paraphrase, shorten, or rewrite the question — copy the exact body text.",
        "",
        "-- EASY (freshers / interns) ----",
    ]
    for q in EASY:
        lines.append(f"  [{q['id']}] {q['title']}  |  tags: {', '.join(q['tags'])}")

    lines.append("\n-- MEDIUM (1-3 years experience) ----")
    for q in MEDIUM:
        lines.append(f"  [{q['id']}] {q['title']}  |  tags: {', '.join(q['tags'])}")

    lines.append("\n-- HARD (senior / 3+ years) ---------")
    for q in HARD:
        lines.append(f"  [{q['id']}] {q['title']}  |  tags: {', '.join(q['tags'])}")

    lines.append("\n─────────────────────────────────────────")
    lines.append("To present a question, call show_coding_question(question_text=<exact body text>).")
    lines.append("─────────────────────────────────────────\n")
    return "\n".join(lines)
