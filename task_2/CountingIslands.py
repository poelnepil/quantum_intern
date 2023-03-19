# You have a matrix MxN that represents a map.
# There are 2 possible states on the map: 1 - islands, 0 - ocean.
# Your task is to calculate the number of islands in the most effective way.
# Please write code in Python 3.

from collections import deque

n = int(input())
m = int(input())

matrix = [[i for i in input().split()] for _ in range(n)]

def how_many_islands(map):

    def bfs(r, c):

        q = deque()
        visited.add((r, c))
        q.append((r, c))

        while q:
            adj_row, adj_col = q.popleft()
            directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

            for direc_row, direc_col in directions:
                r, c = adj_row + direc_row, adj_col + direc_col

                if (r in range(rows)
                and c in range(cols)
                and map[r][c] == '1'
                and (r, c) not in visited):
                    q.append((r, c))
                    visited.add((r, c))

    if not map:
        return 0

    rows, cols = len(map), len(map[0])
    islsnds = 0
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if map[r][c] == '1' and (r, c) not in visited:
                bfs(r, c)
                islsnds += 1

    return islsnds

if __name__ == '__main__':
    print(how_many_islands(matrix))


