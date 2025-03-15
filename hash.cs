using Godot;
using Godot.NativeInterop;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Threading;
using System.Threading.Tasks;
//2025.3.2
public partial class Ground : StaticBody2D
{
    int[] hash_values;
    List<List<int>> hash_values_neighbors = new List<List<int>>();
    float grid_cell_size;
    float grid_cell_inverse;
    int grid_cell_nums_x, grid_cell_nums_y;
    int[] hash_values_st, hash_values_ed;
    int big_num;

    public struct HashValueIndexBind
    {
        public int hash_value { get; set; }
        public int index { get; set; }        
    }
    HashValueIndexBind[] hash_values_sorted;
    
    public void HashInit()
    {
        hash_values = new int[ball_nums_];
        grid_cell_size = 2 * smoothing_length;
        grid_cell_inverse = 1 / grid_cell_size;
        grid_cell_nums_x = (int)Math.Ceiling(GetViewportRect().Size.X * grid_cell_inverse);

        grid_cell_nums_y = (int)Math.Ceiling(GetViewportRect().Size.Y * grid_cell_inverse);
        int grid_cell_nums = grid_cell_nums_x * grid_cell_nums_y;
        big_num = 10 * grid_cell_nums;
        hash_values_st = new int[grid_cell_nums];
        hash_values_ed = new int[grid_cell_nums];
        //Fill hash_values_st with extreme big values at the begin.
        Array.Fill(hash_values, big_num);
        for (int hash_value = 0; hash_value < grid_cell_nums; hash_value++)
        {
            hash_values_neighbors.Add(new List<int>());
            for (int i = 0; i <= 2*grid_cell_nums_x; i+=grid_cell_nums_x)
            {
                for (int j = 0; j < 3; j++)
                {
                    int neighbor_hash_value = hash_value - grid_cell_nums_x - 1 + i + j;
                    if (neighbor_hash_value >= 0 && neighbor_hash_value<=grid_cell_nums - 1)
                        hash_values_neighbors[hash_value].Add(neighbor_hash_value);
                }
            }
        }

    }


    //Use Parallel.for to call this function!
    public void GetHashValue(int index)
    {
        Vector2I ixiy = (Vector2I)(ball_position_[index] * grid_cell_inverse);
        hash_values[index] = ixiy[1] * grid_cell_nums_x + ixiy[0];
    //     GD.Print("ball_position_[index]", ball_position_[index]);
    //     GD.Print("未圆整", ball_position_[index] * grid_cell_inverse);
    //     GD.Print("圆整", (Vector2I)(ball_position_[index] * grid_cell_inverse));

    }

    //
    public void Hashing()
    {
        //Prepare,refill hash_values with big number
        Array.Fill(hash_values_st, big_num);
        Array.Fill(hash_values_ed, big_num);
        //Step1. Calculate hash values of all particles.
        Parallel.For(0, ball_nums_, i =>
        {
            GetHashValue(i);
        });
        //Step2. Bind particel hash value and its index,sort them by hash value.
        hash_values_sorted = hash_values.Select((value, index) =>
                            new HashValueIndexBind { hash_value = value, index = index })
                            .OrderBy(item => item.hash_value)
                            .ToArray();

        //Step3. hash values post processing.
        //a. the smallest hash value start index is exact 0 in hash_values_sorted;
        //b. the biggest hash value end index is exact last index in hash_values_sorted;
        // GD.Print("hash_values_sorted[0].hash_value",hash_values_sorted[0].hash_value);
        // GD.Print("hash_values_sorted[ball_nums_ - 1].hash_value",hash_values_sorted[ball_nums_ - 1].hash_value);
        // GD.Print(hash_values_st.Length);
        hash_values_st[hash_values_sorted[0].hash_value] = 0;
        hash_values_ed[hash_values_sorted[ball_nums_ - 1].hash_value] = ball_nums_ - 1;
        for (int i = 1; i < ball_nums_; i++)
        {
            if (hash_values_sorted[i].hash_value != hash_values_sorted[i - 1].hash_value)
            {
                hash_values_st[hash_values_sorted[i].hash_value] = i;
                hash_values_ed[hash_values_sorted[i - 1].hash_value] = i - 1;
            }
        }
    }

    public List<int> NeighborhoodSearch(int index)
    {
        List<int> neighbor_idx_list = new List<int>();
        //Step1. get the neignbor particle indexes according to index
        var neignbors = hash_values_neighbors[hash_values[index]];
        // GD.Print("hash_values[index]: ", hash_values[index]);
        // for (int i = 0; i < neignbors.Count; i++)
        // {
        //     GD.Print("neignbors ", neignbors[i]);
        // }
        // GD.Print("----");
        for (int i = 0; i < neignbors.Count; i++)
        {
            var ith_neighbor = neignbors[i];
            for (int j = hash_values_st[ith_neighbor]; j <= hash_values_ed[ith_neighbor]; j++)
            {
                if (j != big_num)
                    neighbor_idx_list.Add(hash_values_sorted[j].index);
            }
        }
        
        //  GD.Print("index", index);
        //  GD.Print("neighbor nums", neighbor_idx_list.Count);
        //  for (int i = 0; i < neighbor_idx_list.Count; i++)
        //  {
        //      GD.Print("neighbor ", i, " index : ", neighbor_idx_list[i]);
        //  }
        return neighbor_idx_list;
    }
}