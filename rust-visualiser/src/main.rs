extern crate byteorder;

use std::fs::File;
use std::io::Read;
use byteorder::{ReadBytesExt, LittleEndian};


fn main() {
    println!("Starting Visualising");

    // open file
    let default = "../grid_output.bin";
    let large = "../16384x16384.bin";
    let medium = "../8192x8192.bin";
    let mut file = File::open(default).unwrap();

    // calculate grid size
    let grid_width = read_i32(&mut file) as u32;
    let grid_height = read_i32(&mut file) as u32;
    let grid_size = (grid_height * grid_width) as usize;

    // init grid
    let mut temp = Vec::with_capacity(grid_size);
    temp.resize(grid_size, 0);
    let mut grid_values: Box::<[i32]> = temp.into_boxed_slice();

    // read values from file into RAM
    println!("reading grid values");
    // for i in 0..grid_size {
    //     grid_values[i] = read_i32(&mut file);
    // }
    file.read_i32_into::<LittleEndian>(&mut grid_values).unwrap();

    save_as_image(&grid_values, grid_width, grid_height, 2);

    println!("done");
}

fn save_as_image(grid_values: &[i32], grid_width: u32, grid_height: u32, reduced_resolution_scale: u32) {
    println!("constructing image!");
    let mut image = image::ImageBuffer::new(grid_width / reduced_resolution_scale, grid_height / reduced_resolution_scale);

    // set the rgb value for each pixel
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        // get the max tick_id belonging to this pixel
        let tick_id = (0..reduced_resolution_scale).fold(-1,
            |accumulator_y, dy| max(accumulator_y, (0..reduced_resolution_scale).fold(-1,
                |accumulator_x, dx| max(accumulator_x, grid_values[(dx + x * reduced_resolution_scale + (dy + reduced_resolution_scale * y) * grid_width) as usize])
            ))
        ) as i64;

        *pixel = if tick_id < 0 {
            image::Rgb([0, 0, 0])
        }
        else {
            // map tick_id to rgb
            let quotient = 100_000_000;
            let base = tick_id + quotient / 1_000;
            let r = map_u8(base * 2633 / quotient);
            let g = map_u8(base * 4783 / quotient);
            let b = map_u8(base * 7451 / quotient);

            image::Rgb([r, g, b])
        };
    }

    println!("saving!");
    image.save("image.png").unwrap();
}

// reads a single i32 from the file
fn read_i32(file: &mut File) -> i32 {
    file.read_i32::<LittleEndian>().unwrap()
}

// returns the biggest one of val1 and val2
fn max(val1: i32, val2: i32) -> i32 {
    if val1 > val2 {
        val1
    }
    else {
        val2
    }
}

// special mapping from i64 to u64
fn map_u8(val: i64) -> u8 {
    let modulo = val % 511;
    
    if modulo >= 256 {
        (511 - modulo) as u8
    }
    else {
        modulo as u8
    }
}