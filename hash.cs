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
    Vector2I[] grid_cell_coord;
    Vector2I[] grid_cell_neighbor_offsets;
    float grid_cell_size;
    float grid_cell_inverse;
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
        grid_cell_coord = new Vector2I[ball_nums_];
        grid_cell_size = 2 * smoothing_length;
        grid_cell_inverse = 1 / grid_cell_size;
        big_num = 10 * ball_nums_;
        hash_values_st = new int[ball_nums_];
        hash_values_ed = new int[ball_nums_];
        grid_cell_neighbor_offsets = new Vector2I[9];
        for (int offset_y = 0; offset_y < 3; offset_y++)
        {
            for (int offset_x = 0; offset_x < 3; offset_x++)
            {
                grid_cell_neighbor_offsets[offset_y * 3 + offset_x][0] = offset_x - 1;
                grid_cell_neighbor_offsets[offset_y * 3 + offset_x][1] = offset_y - 1;
            }
        }
  

    }




    //Matthias Muller Spatial Hashing function
    public void GetHashValue(int index)
    {
        grid_cell_coord[index] = (Vector2I)(ball_position_[index] * grid_cell_inverse);
        hash_values[index] = Math.Abs((grid_cell_coord[index][0] * 1789) ^
                                    (grid_cell_coord[index][1] * 31))
                                    % ball_nums_;
    }
    public void Hashing()
    {
        //Prepare,refill hash_values with big number
        Array.Fill(hash_values_st, big_num);
        Array.Fill(hash_values_ed, big_num);
        //Step1. Calculate hash values of all particles.
        Parallel.For(0, ball_nums_, i =>
        {
            // GetHashValue(i);
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
        for (int i = 0; i < grid_cell_neighbor_offsets.Length; i++)
        {
            Vector2I neighbor_grid_cell_coord = grid_cell_coord[index] + grid_cell_neighbor_offsets[i];
            int neighbor_hash_value = Math.Abs((neighbor_grid_cell_coord[0] * 1789) ^
                                            (neighbor_grid_cell_coord[1] * 31))
                                            % ball_nums_;
            for (int j = hash_values_st[neighbor_hash_value]; j <= hash_values_ed[neighbor_hash_value]; j++)
            {
                if (j != big_num)
                {
                    if ((ball_position_[hash_values_sorted[j].index] - ball_position_[index]).Length()
                        < grid_cell_size)
                    {
                        neighbor_idx_list.Add(hash_values_sorted[j].index);
                    }

                }
            }

        }
        return neighbor_idx_list;
    }
}