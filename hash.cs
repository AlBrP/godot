using Godot;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class Ground : StaticBody2D
{
    private int[] hash_values;
    private Vector2I[] grid_cell_coord;
    private Vector2I[] grid_cell_neighbor_offsets;
    private float grid_cell_size;
    private float grid_cell_inverse;
    private int[] hash_values_st, hash_values_ed;
    private int big_num;

    // 方案3：预分配数组，避免每帧分配
    private int[] sorted_indices;
    private int[] sorted_hashes;

    // 自定义比较器
    private sealed class HashComparer : IComparer<int>
    {
        private readonly int[] hashes;
        public HashComparer(int[] hashes) { this.hashes = hashes; }
        public int Compare(int a, int b) => hashes[a].CompareTo(hashes[b]);
    }

    private HashComparer hash_comparer;

    public void HashInit()
    {
        hash_values = new int[ball_nums_];
        grid_cell_coord = new Vector2I[ball_nums_];
        grid_cell_size = 2 * smoothing_length;
        grid_cell_inverse = 1f / grid_cell_size;
        big_num = 10 * ball_nums_;
        hash_values_st = new int[ball_nums_];
        hash_values_ed = new int[ball_nums_];

        // 预分配排序数组
        sorted_indices = new int[ball_nums_];
        sorted_hashes = new int[ball_nums_];
        hash_comparer = new HashComparer(sorted_hashes);

        grid_cell_neighbor_offsets = new Vector2I[9];
        for (int offset_y = 0; offset_y < 3; offset_y++)
        {
            for (int offset_x = 0; offset_x < 3; offset_x++)
            {
                grid_cell_neighbor_offsets[offset_y * 3 + offset_x] = new Vector2I(offset_x - 1, offset_y - 1);
            }
        }
    }

    private void GetHashValue(int index)
    {
        grid_cell_coord[index] = (Vector2I)(pos_[index] * grid_cell_inverse);
        hash_values[index] = Math.Abs((grid_cell_coord[index].X * 1789) ^ (grid_cell_coord[index].Y * 31)) % ball_nums_;
    }

    public void Hashing()
    {
        Array.Fill(hash_values_st, big_num);
        Array.Fill(hash_values_ed, big_num);

        // 并行计算哈希值
        Parallel.For(0, ball_nums_, i => GetHashValue(i));

        // 方案3：Array.Sort 替代 LINQ
        // 1. 复制哈希值到 sorted_hashes
        // 2. 初始化索引数组
        // 3. 根据哈希值排序索引
        Array.Copy(hash_values, sorted_hashes, ball_nums_);
        for (int i = 0; i < ball_nums_; i++) sorted_indices[i] = i;

        Array.Sort(sorted_indices, hash_comparer);

        // 后处理：构建 start/end 表
        int first_hash = hash_values[sorted_indices[0]];
        int last_hash = hash_values[sorted_indices[ball_nums_ - 1]];
        hash_values_st[first_hash] = 0;
        hash_values_ed[last_hash] = ball_nums_ - 1;

        for (int i = 1; i < ball_nums_; i++)
        {
            int curr_hash = hash_values[sorted_indices[i]];
            int prev_hash = hash_values[sorted_indices[i - 1]];
            if (curr_hash != prev_hash)
            {
                hash_values_st[curr_hash] = i;
                hash_values_ed[prev_hash] = i - 1;
            }
        }
    }

    // 方案4：数组替代List，避免GC
    public void SearchNeighborsToCache(int index, ref int[] cache, ref int count)
    {
        float cell_size_sq = grid_cell_size * grid_cell_size;
        count = 0;
        for (int i = 0; i < grid_cell_neighbor_offsets.Length; i++)
        {
            Vector2I neighbor_cell = grid_cell_coord[index] + grid_cell_neighbor_offsets[i];
            int neighbor_hash = Math.Abs((neighbor_cell.X * 1789) ^ (neighbor_cell.Y * 31)) % ball_nums_;

            int start = hash_values_st[neighbor_hash];
            int end = hash_values_ed[neighbor_hash];

            if (start == big_num) continue;

            for (int j = start; j <= end; j++)
            {
                int neighbor_idx = sorted_indices[j];
                Vector2 diff = pos_[neighbor_idx] - pos_[index];
                if (diff.LengthSquared() < cell_size_sq)
                {
                    if (count < MAX_NEIGHBORS)
                    {
                        cache[count++] = neighbor_idx;
                    }
                }
            }
        }
    }
}
