// main.rs - Rust translation of the core parallel block matching logic
// Requires: rayon = "1.5", clap = { version = "4.0", features = ["derive"] }

use clap::Parser;
use colored::Colorize;
use pathfinding::prelude::{Matrix, kuhn_munkres};
use plotters::prelude::*;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Clone)]
struct SpectrumData {
    mz: Vec<f64>,
    intensity: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Block {
    feature_id: String,
    pepmass: f64,
    rtinseconds: f64,
    precursor_charge: i32,
    collision_energy: String,
    other_lines: Vec<String>,
    spectrum: SpectrumData,
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input MGF file
    #[arg(long)]
    input_mgf: String,
    /// M delta for the isotope used
    #[arg(long, default_value_t = 1.00335484)]
    isotope_mz_diff: f64,
    /// maximum difference in retention time (in minutes)
    #[arg(long, default_value_t = 0.05)]
    max_rt_diff: f64,
    /// maximum difference in precursor m/z (in ppm)
    #[arg(long, default_value_t = 5.0)]
    precursor_mz_dev: f64,
    /// minimum number of carbons in the molecule
    #[arg(long, default_value_t = 2)]
    min_carbons: i32,
    /// maximum number of carbons in the molecule
    #[arg(long, default_value_t = 70)]
    max_carbons: i32,
    /// output MGF file suffix
    #[arg(long, default_value_t = String::from("_matchedCleaned"))]
    output_suffix: String,
    /// minimum relative intensity threshold
    #[arg(long, default_value_t = 0.01)]
    min_relative_fragment_intensity: f64,
    /// Output folder for plots
    #[arg(long, default_value_t = String::from("./output_plots"))]
    output_folder: String,
}

impl Block {
    fn print(&self) {
        println!("Feature ID: {}", self.feature_id);
        println!("Pepmass: {}", self.pepmass);
        println!("RT in seconds: {}", self.rtinseconds);
        println!("Precursor Charge: {}", self.precursor_charge);
        println!("Collision Energy: {}", self.collision_energy);
        println!("Spectrum Data (m/z and intensity):");
        let mz_len = self.spectrum.mz.len();
        let intensity_len = self.spectrum.intensity.len();
        if mz_len > 0 && intensity_len > 0 {
            for i in 0..3.min(mz_len) {
                println!("  m/z: {}, intensity: {}", self.spectrum.mz[i], self.spectrum.intensity[i]);
            }
            if mz_len > 3 {
                println!("  ...");
                println!("  m/z: {}, intensity: {}", self.spectrum.mz[mz_len - 1], self.spectrum.intensity[intensity_len - 1]);
            }
        } else {
            println!("  No spectrum data available.");
        }
    }
}

fn parse_mgf<P: AsRef<Path>>(path: P) -> Vec<Block> {
    let file = File::open(path).expect("Cannot open file");
    let reader = BufReader::new(file);
    let mut blocks = Vec::new();
    let mut feature_id = String::new();
    let mut pepmass = 0.0;
    let mut rtinseconds = 0.0;
    let mut collision_energy = String::new();
    let mut other_lines = Vec::new();
    let mut mz = Vec::new();
    let mut intensity = Vec::new();

    for line in reader.lines() {
        let l = line.unwrap();
        if l.starts_with("FEATURE_ID=") {
            feature_id = l[11..].to_string();
        } else if l.starts_with("PEPMASS=") {
            pepmass = l[8..].split_whitespace().next().unwrap().parse().unwrap();
        } else if l.starts_with("RTINSECONDS=") {
            rtinseconds = l[12..].parse().unwrap();
        } else if l.starts_with("COLLISION_ENERGY=") {
            let key_value: Vec<&str> = l.splitn(2, '=').collect();
            if key_value.len() == 2 {
                let value = key_value[1].trim().to_string();
                if value.starts_with('[') && value.ends_with(']') {
                    // Already a list, normalize whitespace and sort
                    let mut items: Vec<f64> = value[1..value.len() - 1].split(',').filter_map(|v| v.trim().parse().ok()).collect();
                    items.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    collision_energy = format!("{:?}", items).replace(" ", "");
                } else if value.contains(',') {
                    // Comma-separated list, parse, sort, and format as Rust list string
                    let mut items: Vec<f64> = value.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                    items.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    collision_energy = format!("{:?}", items).replace(" ", "");
                } else {
                    // Single value, wrap in list if it's a number
                    if let Ok(num) = value.parse::<f64>() {
                        collision_energy = format!("[{}]", num);
                    }
                }
            }
        } else if l == "BEGIN IONS" {
            feature_id = String::new();
            pepmass = 0.0;
            rtinseconds = 0.0;
            collision_energy = String::new();
            mz.clear();
            intensity.clear();
            other_lines.clear();
        } else if l.starts_with("Num peaks") {
        } else if l == "END IONS" {
            blocks.push(Block {
                feature_id: feature_id.clone(),
                pepmass: pepmass,
                rtinseconds: rtinseconds,
                precursor_charge: 1, // TODO change, for now: Assuming precursor_charge is always 1 for simplicity
                collision_energy: collision_energy.clone(),
                other_lines: other_lines.clone(),
                spectrum: SpectrumData {
                    mz: mz.clone(),
                    intensity: intensity.clone(),
                },
            });
        } else if l.contains(' ') && !l.contains('=') {
            let parts: Vec<&str> = l.split_whitespace().collect();
            if parts.len() == 2 {
                mz.push(parts[0].parse().unwrap());
                intensity.push(parts[1].parse().unwrap());
            }
        } else if l.contains('=') {
            let key_value: Vec<&str> = l.splitn(2, '=').collect();
            if key_value.len() == 2 {
                let key = key_value[0].trim();
                let value = key_value[1].trim().to_string();
                other_lines.push(format!("{}={}", key, value));
            }
        } else if l == "" {
        } else {
            panic!("Unhandled line format: {}", l);
        }
    }
    blocks
}

fn export_mgf<P: AsRef<Path>>(blocks: &[Block], output_path: P) {
    let file = File::create(output_path).expect("Cannot create file");
    let mut writer = std::io::BufWriter::new(file);

    for block in blocks {
        writeln!(writer, "BEGIN IONS").unwrap();
        if !block.feature_id.is_empty() {
            writeln!(writer, "FEATURE_ID={}", block.feature_id).unwrap();
        }
        writeln!(writer, "PEPMASS={}", block.pepmass).unwrap();
        writeln!(writer, "RTINSECONDS={}", block.rtinseconds).unwrap();
        if !block.collision_energy.is_empty() {
            writeln!(writer, "COLLISION_ENERGY={}", block.collision_energy).unwrap();
        }
        for line in &block.other_lines {
            writeln!(writer, "{}", line).unwrap();
        }
        writeln!(writer, "Num peaks={}", block.spectrum.mz.len()).unwrap();
        for (mz, intensity) in block.spectrum.mz.iter().zip(&block.spectrum.intensity) {
            writeln!(writer, "{} {}", mz, intensity).unwrap();
        }
        writeln!(writer, "END IONS").unwrap();
        writeln!(writer, "").unwrap();
    }
}

fn filter_blocks_by_relative_intensity(blocks: &mut Vec<Block>, min_relative_intensity: f64) {
    for block in blocks.iter_mut() {
        if let Some(&max_intensity) = block.spectrum.intensity.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            let filtered_mz_intensity: Vec<(f64, f64)> = block
                .spectrum
                .mz
                .iter()
                .zip(&block.spectrum.intensity)
                .filter(|&(_, &intensity)| intensity / max_intensity >= min_relative_intensity)
                .map(|(&mz, &intensity)| (mz, intensity))
                .collect();

            block.spectrum = SpectrumData {
                mz: filtered_mz_intensity.iter().map(|&(mz, _)| mz).collect(),
                intensity: filtered_mz_intensity.iter().map(|&(_, intensity)| intensity).collect(),
            };
        }
    }
}

fn normalize_spectra_intensity(blocks: &mut Vec<Block>) {
    for block in blocks.iter_mut() {
        let total_intensity: f64 = block.spectrum.intensity.iter().sum();
        if total_intensity > 0.0 {
            block.spectrum.intensity.iter_mut().for_each(|intensity| {
                *intensity /= total_intensity;
            });
        }
    }
}

fn cosine_similarity(spec: &Block, spec_other: &Block, isotope_mz_diff: f64, min_carbons: i32, max_carbons: i32, fragment_mz_tolerance: f64, direction: i32) -> (f64, Vec<i32>, Vec<i32>) {
    if spec.spectrum.mz.len() > spec_other.spectrum.mz.len() {
        let (score, assignments_a, assignments_b) = cosine_similarity(spec_other, spec, isotope_mz_diff, min_carbons, max_carbons, fragment_mz_tolerance, direction * -1);
        return (score, assignments_b, assignments_a);
    }

    if spec.spectrum.mz.len() == 0 || spec_other.spectrum.mz.len() == 0 {
        println!("Warning: One of the spectra has no fragments, returning 0.0");
        return (0.0, Vec::new(), Vec::new());
    }

    let mut cost_matrix_intensity = vec![vec![0; spec_other.spectrum.mz.len() * 2]; spec.spectrum.mz.len()];
    for (peak_index, (&peak_mz, &peak_intensity)) in spec.spectrum.mz.iter().zip(&spec.spectrum.intensity).enumerate() {
        for (peak_other_index, (&peak_other_mz, &peak_other_intensity)) in spec_other.spectrum.mz.iter().zip(&spec_other.spectrum.intensity).enumerate() {
            for xn in min_carbons..max_carbons + 1 {
                let peak_adjusted_mz = peak_mz + (isotope_mz_diff * (direction as f64)) * xn as f64;
                if (peak_adjusted_mz - peak_other_mz).abs() <= fragment_mz_tolerance {
                    cost_matrix_intensity[peak_index][peak_other_index] = (1. + peak_intensity * peak_other_intensity * 1_000_000.0) as i64;
                }
            }
            cost_matrix_intensity[peak_index][spec_other.spectrum.mz.len() + peak_other_index] = (1) as i64;
        }
    }

    if false {
        //Print cost matrix
        if spec.spectrum.mz.len() < 20 && spec_other.spectrum.mz.len() < 20 {
            println!("Cost matrix intensity:");
            print!("         |");
            for (_, mz) in spec_other.spectrum.mz.iter().enumerate() {
                print!("{:>8.4} ", mz);
            }
            println!("");
            print!("---------+");
            for (_, _) in spec_other.spectrum.mz.iter().enumerate() {
                print!("---------");
            }
            println!("");
            for (rowi, _) in spec.spectrum.mz.iter().enumerate() {
                print!("{:>8.4} |", spec.spectrum.mz[rowi]);
                for (coli, _) in spec_other.spectrum.mz.iter().enumerate() {
                    let value = cost_matrix_intensity[rowi][coli];
                    if value > 0 {
                        print!("{:>8.4} ", value);
                    } else {
                        print!("         ");
                    }
                }
                println!("");
            }
        }
    }

    let weights = Matrix::from_rows(cost_matrix_intensity).unwrap();

    let (score, assignments) = kuhn_munkres(&weights);

    let adjusted_assignments_a: Vec<i32> = assignments.iter().map(|&a| if a < spec_other.spectrum.mz.len() { a as i32 } else { -1 }).collect();

    let mut adjusted_assignments_b = Vec::with_capacity(spec_other.spectrum.mz.len());
    for b_idx in 0..spec_other.spectrum.mz.len() {
        let a_idx = adjusted_assignments_a.iter().position(|&a| a == b_idx as i32);
        if let Some(idx) = a_idx {
            adjusted_assignments_b.push(idx as i32);
        } else {
            adjusted_assignments_b.push(-1);
        }
    }

    //TODO implement score check and optimization

    return (score as f64 / 1_000_000.0, adjusted_assignments_a, adjusted_assignments_b);
}

fn isotopolog_match_optimization(block_a: &Block, block_b: &Block, isotope_mz_diff: f64, max_mz_dev_ppm: f64, min_carbons: i32, max_carbons: i32) -> (f64, Vec<i32>, Vec<i32>) {
    let (c_score, c_assignment_a, c_assignment_b) = cosine_similarity(block_a, block_b, isotope_mz_diff, min_carbons, max_carbons, 0.005, 1);

    return (c_score, c_assignment_a, c_assignment_b);
}

fn draw_fragments_table(area: &plotters::drawing::DrawingArea<plotters_svg::SVGBackend, plotters::coord::Shift>, table_data: &[(f64, f64, i32, f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    area.fill(&WHITE)?;

    let title = "Matched Fragments Table";
    let headers = ["Native m/z", "Labeled m/z", "Carbon Atoms", "Native Rel. Int. (%)", "Labeled Rel. Int. (%)"];

    let font_size = 12;
    let header_font_size = 14;
    let title_font_size = 18;
    let row_height = 22;
    let available_width = area.dim_in_pixel().0 as i32 - 40; // Leave 20px margin on each side
    let col_width = available_width / headers.len() as i32;
    let start_x = 20;
    let start_y = 30;

    // Separate and sort data: high abundance (>1%) first, then low abundance (≤1%)
    let highly_abundant_threshold = 1.0;
    let mut high_abundance: Vec<_> = table_data
        .iter()
        .filter(|&&(_, _, _, native_rel_int, _)| native_rel_int >= highly_abundant_threshold)
        .cloned()
        .collect();
    let mut low_abundance: Vec<_> = table_data.iter().filter(|&&(_, _, _, native_rel_int, _)| native_rel_int < highly_abundant_threshold).cloned().collect();

    // Sort each group by m/z value (native m/z)
    high_abundance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    low_abundance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Draw title
    let title_x = area.dim_in_pixel().0 as i32 / 2 - (title.len() as i32 * title_font_size / 4);
    area.draw(&Text::new(title, (title_x, start_y), ("sans-serif", title_font_size).into_font().color(&BLACK)))?;

    // Draw headers
    let header_y = start_y + 20;
    for (i, header) in headers.iter().enumerate() {
        let x = start_x + i as i32 * col_width;
        area.draw(&Text::new(*header, (x, header_y), ("sans-serif", header_font_size).into_font().color(&BLACK)))?;

        // Draw header underline (offset below the text)
        area.draw(&PathElement::new(vec![(x, header_y + 15), (x + col_width - 10, header_y + 15)], &BLACK))?;
    }

    // Draw data rows
    let text_vertical_offset = -11; // Configurable offset to move text upwards (negative values move up)
    let mut current_row = 0;

    // Function to draw a section of data
    let mut draw_section = |section_data: &[(f64, f64, i32, f64, f64)], section_title: &str, start_row: &mut i32| -> Result<(), Box<dyn std::error::Error>> {
        // Draw section header if there's data
        if !section_data.is_empty() {
            let section_y = header_y + 40 + *start_row * row_height;
            area.draw(&Text::new(
                section_title,
                (start_x, section_y),
                ("sans-serif", header_font_size).into_font().color(&RGBColor(100, 100, 100)),
            ))?;
            *start_row += 1;

            // Reduced spacing after section header
        }
        for (idx, &(native_mz, labeled_mz, carbon_atoms, native_rel_int, labeled_rel_int)) in section_data.iter().enumerate() {
            let row_top = header_y + 35 + *start_row * row_height;
            let row_bottom = row_top + row_height;
            let text_y = row_top + (row_height / 2) + (font_size / 2) + text_vertical_offset;

            // Add alternating row background (light gray for even rows within each section)
            if idx % 2 == 0 {
                area.draw(&Rectangle::new(
                    [(start_x - 5, row_top - 2), (start_x + headers.len() as i32 * col_width - 15, row_bottom - 2)],
                    ShapeStyle::from(&RGBColor(245, 245, 245)).filled(),
                ))?;
            }

            // Prepare row data
            let row_values = [
                format!("{:.4}", native_mz),
                format!("{:.4}", labeled_mz),
                format!("{}", carbon_atoms),
                format!("{:.2}", native_rel_int),
                format!("{:.2}", labeled_rel_int),
            ];

            for (col_idx, value) in row_values.iter().enumerate() {
                let x = start_x + col_idx as i32 * col_width;
                area.draw(&Text::new(value.as_str(), (x, text_y), ("sans-serif", font_size).into_font().color(&BLACK)))?;
            }
            *start_row += 1;
        }

        // Reduced spacing between sections
        if !section_data.is_empty() {
            // No extra spacing - sections will be closer together
        }

        Ok(())
    };

    // Draw high abundance section using the threshold variable
    let high_abundance_title = format!("High Abundance Fragments (≥{}%)", highly_abundant_threshold);
    draw_section(&high_abundance, &high_abundance_title, &mut current_row)?;
    current_row += 1; // Add extra space between sections

    // Draw low abundance section using the threshold variable
    let low_abundance_title = format!("Low Abundance Fragments (<{}%)", highly_abundant_threshold);
    draw_section(&low_abundance, &low_abundance_title, &mut current_row)?;

    Ok(())
}

fn plot_blocks_to_pdf(block_a: &Block, block_b: &Block, assignment_a: Vec<i32>, assignment_b: Vec<i32>, filename: &str, isotope_mz_diff: f64) -> Result<(), Box<dyn std::error::Error>> {
    // Prepare data for plotting
    let mut mzs = block_a.spectrum.mz.clone();
    mzs.extend(&block_b.spectrum.mz);
    let min_mz = mzs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_mz = mzs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let max_intensity_a = block_a.spectrum.intensity.iter().cloned().fold(0.0, f64::max) * 1.1;
    let max_intensity_b = block_b.spectrum.intensity.iter().cloned().fold(0.0, f64::max) * 1.1;

    let max_intensity = max_intensity_a.max(max_intensity_b);

    let title = format!(
        "Raw spectra - native: m/z {:.4} @ {:.2}s, labeled: m/z {:.4} @ {:.2}s",
        block_a.pepmass, block_a.rtinseconds, block_b.pepmass, block_b.rtinseconds
    );

    // Calculate total intensities for relative intensity calculations
    let total_intensity_a: f64 = block_a.spectrum.intensity.iter().sum();
    let total_intensity_b: f64 = block_b.spectrum.intensity.iter().sum();

    // Prepare table data for matched fragments
    let mut table_data = Vec::new();

    for (i, &assignment) in assignment_a.iter().enumerate() {
        if assignment >= 0 {
            let native_mz = block_a.spectrum.mz[i];
            let native_intensity = block_a.spectrum.intensity[i];
            let labeled_mz = block_b.spectrum.mz[assignment as usize];
            let labeled_intensity = block_b.spectrum.intensity[assignment as usize];

            // Calculate number of carbon atoms from mz difference
            let mz_diff = labeled_mz - native_mz;
            let carbon_atoms = (mz_diff / isotope_mz_diff).round() as i32;

            // Calculate relative intensities
            let relative_intensity_native = (native_intensity / total_intensity_a) * 100.0;
            let relative_intensity_labeled = (labeled_intensity / total_intensity_b) * 100.0;

            table_data.push((native_mz, labeled_mz, carbon_atoms, relative_intensity_native, relative_intensity_labeled));
        }
    }

    let plot_width = 1400; // Increased width for 2-column layout
    let plot_height = 600 * 3; // 3 plots height
    let root = SVGBackend::new(filename, (plot_width, plot_height)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split into 2 columns: plots on left, table on right
    let main_areas = root.split_evenly((1, 2));
    let plots_area = main_areas.get(0).unwrap();
    let table_area = main_areas.get(1).unwrap();

    // Split the plots area into 3 vertical plots
    let plot_areas = plots_area.split_evenly((3, 1));
    let top = plot_areas.get(0).unwrap();
    let middle = plot_areas.get(1).unwrap();
    let bottom = plot_areas.get(2).unwrap();

    let mut chart = ChartBuilder::on(top)
        .caption(title, ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_mz + 15.0, -max_intensity..max_intensity)?;

    chart.configure_mesh().x_desc("m/z").y_desc("Intensity (native positive, labeled negative)").draw()?;

    // Plot block A (positive intensities)
    for (&mz, &intensity) in block_a.spectrum.mz.iter().zip(&block_a.spectrum.intensity) {
        chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, intensity)], &BLUE)))?;
        // Add m/z label above the peak
        let label_y = intensity + max_intensity * 0.02;
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.4}", mz),
            (mz, label_y),
            ("sans-serif", 3).into_font().color(&RGBColor(128, 128, 128)),
        )))?;
    }

    // Plot block B (negative intensities)
    for (&mz, &intensity) in block_b.spectrum.mz.iter().zip(&block_b.spectrum.intensity) {
        chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, -intensity)], &RED)))?;
        // Add m/z label below the negative peak
        let label_y = -intensity - max_intensity * 0.02;
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.4}", mz),
            (mz, label_y),
            ("sans-serif", 3).into_font().color(&RGBColor(128, 128, 128)),
        )))?;
    }

    let mut chart = ChartBuilder::on(middle)
        .caption("Raw spectra with matching peaks highlighted", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_mz + 15.0, -max_intensity..max_intensity)?;

    chart.configure_mesh().x_desc("m/z").y_desc("Intensity (native positive, labeled negative)").draw()?;

    // Plot block A (positive intensities)
    for (i, (&mz, &intensity)) in block_a.spectrum.mz.iter().zip(&block_a.spectrum.intensity).enumerate() {
        if assignment_a[i] < 0 {
            chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, intensity)], &BLUE)))?;
        } else {
            chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, intensity)], ShapeStyle::from(&BLUE).stroke_width(3))))?;
        }
        // Add m/z label above the peak
        let label_y = intensity + max_intensity * 0.02;
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.4}", mz),
            (mz, label_y),
            ("sans-serif", 3).into_font().color(&RGBColor(128, 128, 128)),
        )))?;
    }

    // Plot block B (negative intensities)
    for (i, (&mz, &intensity)) in block_b.spectrum.mz.iter().zip(&block_b.spectrum.intensity).enumerate() {
        if assignment_b[i] < 0 {
            chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, -intensity)], &RED)))?;
        } else {
            chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, -intensity)], ShapeStyle::from(&RED).stroke_width(3))))?;
        }
        // Add m/z label below the negative peak
        let label_y = -intensity - max_intensity * 0.02;
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.4}", mz),
            (mz, label_y),
            ("sans-serif", 3).into_font().color(&RGBColor(128, 128, 128)),
        )))?;
    }

    let mut chart = ChartBuilder::on(bottom)
        .caption("Cleaned, native spectrum", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_mz + 15.0, 0.0..max_intensity_a)?;

    chart.configure_mesh().x_desc("m/z").y_desc("Intensity (cleaned native)").draw()?;

    // Plot block A (positive intensities)
    for (i, (&mz, &intensity)) in block_a.spectrum.mz.iter().zip(&block_a.spectrum.intensity).enumerate() {
        if assignment_a[i] >= 0 {
            chart.draw_series(std::iter::once(PathElement::new(vec![(mz, 0.0), (mz, intensity)], &BLUE)))?;

            // Calculate carbon atoms for this matched fragment
            let labeled_mz = block_b.spectrum.mz[assignment_a[i] as usize];
            let mz_diff = labeled_mz - mz;
            let carbon_atoms = (mz_diff / isotope_mz_diff).round() as i32;

            // Add m/z and carbon count label above the peak
            let label_y = intensity + max_intensity_a * 0.02;
            chart.draw_series(std::iter::once(Text::new(
                format!("{:.4};C{}", mz, carbon_atoms),
                (mz, label_y),
                ("sans-serif", 3).into_font().color(&RGBColor(128, 128, 128)),
            )))?;
        }
    }

    // Draw the table
    draw_fragments_table(table_area, &table_data)?;

    root.present()?;
    Ok(())
}

fn find_block_pairs(blocks: &Vec<Block>, isotope_mz_diff: f64, max_rt_diff: f64, precursor_mz_dev: f64, min_carbons: i32, max_carbons: i32, output_folder: &str) -> Vec<(Block, Block, Block)> {
    let results = Mutex::new(Vec::new());
    let mut matching_blocks = 1;

    let output_folder = output_folder.to_string();
    let output_path = Path::new(&output_folder);
    if !output_path.exists() {
        if let Err(e) = fs::create_dir_all(&output_path) {
            panic!("Failed to create output folder {}: {}", output_folder, e);
        }
    }

    for (i, block1) in blocks.iter().enumerate() {
        let curResults = Mutex::new(Vec::new());
        let pepmass1 = block1.pepmass;
        let rt1 = block1.rtinseconds / 60.0;

        println!("Processing block {}: feature_id={}, precursor mz={}, rt={} min ", i + 1, block1.feature_id, pepmass1, rt1);

        for (j, block2) in blocks.iter().enumerate() {
            let pepmass2 = block2.pepmass;
            let rt2 = block2.rtinseconds / 60.0;
            if i == j || (rt1 - rt2).abs() > max_rt_diff || pepmass1 > pepmass2 || pepmass1 + min_carbons as f64 > pepmass2 {
                continue;
            }

            let mz_diff = (pepmass2 - pepmass1).abs();
            let est_xn = (mz_diff / isotope_mz_diff).round() as i32;
            let accounted_mz_difference = ((mz_diff - est_xn as f64 * isotope_mz_diff) / pepmass1 * 1e6).abs();

            if est_xn >= min_carbons && est_xn <= max_carbons && accounted_mz_difference <= precursor_mz_dev {
                matching_blocks += 1;
                println!(
                    "   - Found {}-ith matching pair\n      - Feature_id {}\n      - precursor mz: {}\n      - rt: {} min\n      - {} and {} fragment peaks\n      - est_xn: {}\n      - rt_diff: {:.2} min\n      - mz_difference of precursors after accounting for carbon atoms {:.2} ppm",
                    matching_blocks,
                    block2.feature_id,
                    pepmass2,
                    rt2,
                    block1.spectrum.mz.len(),
                    block2.spectrum.mz.len(),
                    est_xn,
                    (rt1 - rt2).abs(),
                    accounted_mz_difference
                );

                let start = Instant::now();
                let result = std::panic::catch_unwind(|| isotopolog_match_optimization(block1, block2, isotope_mz_diff, precursor_mz_dev, min_carbons, est_xn));

                match result {
                    Ok(res) => {
                        let score = res.0;
                        let assignment_a = res.1;
                        let assignment_b = res.2;

                        let used_fragments_n = assignment_a.iter().filter(|&&a| a >= 0).count();

                        println!("      - Cosine similarity score (took {:.2} sec): {:.2}", start.elapsed().as_secs_f64(), score);
                        println!(
                            "      - using {} fragments of the {} fragments of spectrum A and {} fragments of spectrum B",
                            used_fragments_n,
                            block1.spectrum.mz.len(),
                            block2.spectrum.mz.len()
                        );

                        plot_blocks_to_pdf(
                            block1,
                            block2,
                            assignment_a.clone(),
                            assignment_b.clone(),
                            &format!("{}/output_{}_{}.svg", output_folder, block1.feature_id, block2.feature_id),
                            isotope_mz_diff,
                        )
                        .unwrap_or_else(|e| {
                            println!("{}", format!("      - Error plotting blocks to PDF: {}", e).red());
                        });
                        println!("      - Exported plot to {}/output_{}_{}.svg", output_folder, block1.feature_id, block2.feature_id);

                        let mut cleaned_block1 = block1.clone();
                        let mut new_mz = Vec::new();
                        let mut new_intensity = Vec::new();
                        for (idx, &assign) in assignment_a.iter().enumerate() {
                            if assign >= 0 {
                                new_mz.push(cleaned_block1.spectrum.mz[idx]);
                                new_intensity.push(cleaned_block1.spectrum.intensity[idx]);
                            }
                        }
                        cleaned_block1.spectrum.mz = new_mz;
                        cleaned_block1.spectrum.intensity = new_intensity;

                        curResults.lock().unwrap().push((block1.clone(), block2.clone(), cleaned_block1.clone()));
                    }
                    Err(_) => {
                        println!("   - Error occurred during isotopolog_match_optimization, skipping...");
                    }
                }
            }
        }
        if curResults.lock().unwrap().len() == 0 {
            println!("{}", format!("   - Warning: No matching other spectra found.").yellow());
        } else if curResults.lock().unwrap().len() > 1 {
            println!("{}", format!("   - Error: Found more than 1 matching spectra, skipping for now.").red());
        } else {
            results.lock().unwrap().extend(curResults.into_inner().unwrap());
        }
        println!("");
    }
    results.into_inner().unwrap()
}

fn main() {
    let args = Args::parse();

    println!("---------------------------------------------------------------------");
    println!("Input MGF file: {}", args.input_mgf);
    let mut blocks = parse_mgf(&args.input_mgf);
    println!("Isotope m/z difference: {}", args.isotope_mz_diff);
    println!("Maximum retention time difference (minutes): {}", args.max_rt_diff);
    println!("Maximum precursor m/z deviation (ppm): {}", args.precursor_mz_dev);
    println!("Minimum number of carbons: {}", args.min_carbons);
    println!("Maximum number of carbons: {}", args.max_carbons);
    println!("Output MGF file suffix: {}", args.output_suffix);
    println!("");

    println!("---------------------------------------------------------------------");
    println!("Parsed {} blocks, one example is:", blocks.len());
    if let Some(block) = blocks.get(0) {
        block.print();
    } else {
        println!("No blocks found.");
        return;
    }
    println!("");

    println!("---------------------------------------------------------------------");
    println!("Filtering spectra for fragments above threshold of {}...", args.min_relative_fragment_intensity);
    filter_blocks_by_relative_intensity(&mut blocks, args.min_relative_fragment_intensity);
    println!("\n");

    println!("---------------------------------------------------------------------");
    println!("Normalizing fragment abundances...");
    normalize_spectra_intensity(&mut blocks);
    println!("\n");

    println!("---------------------------------------------------------------------");
    println!("Finding putative matching block pairs...");
    let pairs = find_block_pairs(
        &mut blocks,
        args.isotope_mz_diff,
        args.max_rt_diff,
        args.precursor_mz_dev,
        args.min_carbons,
        args.max_carbons,
        &args.output_folder,
    );
    println!("Found {} putative matching block pairs.", pairs.len());
    println!("");

    println!("---------------------------------------------------------------------");
    println!("Exporting matching block pairs to MGF format...");
    //TODO only use the beset match
    let output_path = {
        let mut path = args.input_mgf.clone();
        if let Some(pos) = path.rfind(".mgf") {
            path.replace_range(pos.., &format!("{}{}", args.output_suffix, ".mgf"));
        } else {
            path.push_str(&format!("{}{}", args.output_suffix, ".mgf"));
        }
        path
    };
    export_mgf(&pairs.iter().map(|(_, _, cleaned_spectrum)| cleaned_spectrum.clone()).collect::<Vec<_>>(), &output_path);
    println!("Exported {} spectra to {}", pairs.len(), output_path);
    println!("")
}
