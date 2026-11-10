use rand::Rng;
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const INPUTS: usize = 36;
const HIDDEN: usize = 40;
const OUTPUTS: usize = 36;
const PATCH: u32 = 6;

struct Perceptron {
    layers: [usize; 3],
    alpha: f64,
    input_hidden: Vec<f64>,
    hidden_output: Vec<f64>,
}

impl Perceptron {
    fn new(alpha: f64) -> Self {
        let mut rng = rand::thread_rng();
        let input_hidden = (0..(INPUTS + 1) * HIDDEN)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        let hidden_output = (0..(HIDDEN + 1) * OUTPUTS)
            .map(|_| rng.gen_range(-0.5..0.5))
            .collect();
        Self {
            layers: [INPUTS, HIDDEN, OUTPUTS],
            alpha,
            input_hidden,
            hidden_output,
        }
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp = value.exp();
            exp / (1.0 + exp)
        }
    }

    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(input.len(), INPUTS);
        let mut hidden = vec![0.0; HIDDEN];
        for (j, value) in hidden.iter_mut().enumerate() {
            let mut sum = self.input_hidden[INPUTS * HIDDEN + j];
            for (i, input_value) in input.iter().enumerate() {
                sum += input_value * self.input_hidden[i * HIDDEN + j];
            }
            *value = Self::sigmoid(sum);
        }

        let mut output = vec![0.0; OUTPUTS];
        for (k, value) in output.iter_mut().enumerate() {
            let mut sum = self.hidden_output[HIDDEN * OUTPUTS + k];
            for (j, hidden_value) in hidden.iter().enumerate() {
                sum += hidden_value * self.hidden_output[j * OUTPUTS + k];
            }
            *value = Self::sigmoid(sum);
        }
        (hidden, output)
    }

    fn process(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).1
    }

    fn train_example(&mut self, input: &[f64], target: &[f64]) -> f64 {
        let (hidden, output) = self.forward(input);
        let difference: Vec<f64> = target.iter().zip(&output).map(|(t, o)| t - o).collect();
        let error: f64 = difference.iter().map(|d| d * d).sum();

        let output_delta: Vec<f64> = difference
            .iter()
            .zip(&output)
            .map(|(d, o)| d * o * (1.0 - o))
            .collect();
        let mut hidden_delta = vec![0.0; HIDDEN];
        for (j, delta) in hidden_delta.iter_mut().enumerate() {
            let downstream: f64 = (0..OUTPUTS)
                .map(|k| self.hidden_output[j * OUTPUTS + k] * output_delta[k])
                .sum();
            *delta = downstream * hidden[j] * (1.0 - hidden[j]);
        }

        for (j, hidden_value) in hidden.iter().enumerate() {
            for (k, output_value) in output_delta.iter().enumerate() {
                self.hidden_output[j * OUTPUTS + k] += self.alpha * hidden_value * output_value;
            }
        }
        for (k, delta) in output_delta.iter().enumerate() {
            self.hidden_output[HIDDEN * OUTPUTS + k] += self.alpha * delta;
        }
        for (i, input_value) in input.iter().enumerate() {
            for (j, delta) in hidden_delta.iter().enumerate() {
                self.input_hidden[i * HIDDEN + j] += self.alpha * input_value * delta;
            }
        }
        for (j, delta) in hidden_delta.iter().enumerate() {
            self.input_hidden[INPUTS * HIDDEN + j] += self.alpha * delta;
        }
        error
    }

    fn to_value(&self) -> Value {
        let mut synapses = Vec::with_capacity((INPUTS + 1) * HIDDEN + (HIDDEN + 1) * OUTPUTS);
        for (layer, (weights, origins, destinations)) in [
            (&self.input_hidden, INPUTS + 1, HIDDEN),
            (&self.hidden_output, HIDDEN + 1, OUTPUTS),
        ]
        .iter()
        .enumerate()
        {
            for origin in 0..*origins {
                for destination in 0..*destinations {
                    synapses.push(json!({
                        "layer": layer,
                        "origin": origin,
                        "destination": destination,
                        "weight": weights[origin * *destinations + destination]
                    }));
                }
            }
        }
        json!({
            "netConfiguration": self.layers,
            "activationFunction": "sigmoid",
            "synapses": synapses
        })
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec(&self.to_value())?)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let payload: Value = serde_json::from_slice(&fs::read(path)?)?;
        if payload["activationFunction"].as_str().unwrap_or("sigmoid") != "sigmoid" {
            return Err("Only sigmoid models are supported".into());
        }
        let config = payload["netConfiguration"].as_array().ok_or("Missing netConfiguration")?;
        if config.len() != 3
            || config[0].as_u64() != Some(INPUTS as u64)
            || config[1].as_u64() != Some(HIDDEN as u64)
            || config[2].as_u64() != Some(OUTPUTS as u64)
        {
            return Err("Model must use the 36-40-36 network configuration".into());
        }
        let mut model = Self {
            layers: [INPUTS, HIDDEN, OUTPUTS],
            alpha: 1.0,
            input_hidden: vec![0.0; (INPUTS + 1) * HIDDEN],
            hidden_output: vec![0.0; (HIDDEN + 1) * OUTPUTS],
        };
        let synapses = payload["synapses"].as_array().ok_or("Missing synapses")?;
        for synapse in synapses {
            let layer = synapse["layer"].as_u64().ok_or("Invalid layer")? as usize;
            let origin = synapse["origin"].as_u64().ok_or("Invalid origin")? as usize;
            let destination = synapse["destination"].as_u64().ok_or("Invalid destination")? as usize;
            let weight = synapse["weight"].as_f64().ok_or("Invalid weight")?;
            if layer >= 2 {
                return Err("Invalid synapse layer".into());
            }
            let width = if layer == 0 { HIDDEN } else { OUTPUTS };
            let matrix = if layer == 0 { &mut model.input_hidden } else { &mut model.hidden_output };
            if origin >= matrix.len() / width || destination >= width {
                return Err("Synapse index is outside the model dimensions".into());
            }
            matrix[origin * width + destination] = weight;
        }
        Ok(model)
    }
}

fn default_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn train(source: &Path, target: &Path, model_path: &Path, steps: usize) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    if steps == 0 {
        return Err("Training steps must be positive".into());
    }
    let source_image = image::open(source)?.to_rgb8();
    let target_image = image::open(target)?.to_rgb8();
    if source_image.dimensions() != target_image.dimensions() {
        return Err("Source and target images must have identical dimensions".into());
    }
    let (width, height) = source_image.dimensions();
    if width <= PATCH || height <= PATCH {
        return Err("Training images must be larger than 6x6 pixels".into());
    }
    let mut rng = rand::thread_rng();
    let coordinates: Vec<(u32, u32)> = (0..steps)
        .map(|_| (rng.gen_range(0..width - PATCH), rng.gen_range(0..height - PATCH)))
        .collect();
    let mut model = Perceptron::new(1.0);

    for (count, (x, y)) in coordinates.into_iter().enumerate() {
        let mut inputs = Vec::with_capacity(INPUTS);
        let mut outputs = Vec::with_capacity(OUTPUTS);
        for dy in 0..PATCH {
            for dx in 0..PATCH {
                inputs.push(source_image.get_pixel(x + dx, y + dy)[2] as f64 / 256.0);
                outputs.push(target_image.get_pixel(x + dx, y + dy)[2] as f64 / 256.0);
            }
        }
        model.train_example(&inputs, &outputs);
        if (count + 1) % 50_000 == 0 {
            println!("Trained cases: {}", count + 1);
        }
    }
    println!("Training time: {:.2} seconds", started.elapsed().as_secs_f64());
    model.save(model_path)?;
    println!("Saved model to {}", model_path.display());
    Ok(())
}

fn convert(image_path: &Path, model_path: &Path, output_path: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let model = Perceptron::load(model_path)?;
    let gray = image::open(image_path)?.to_luma8();
    let (width, height) = gray.dimensions();
    if width < 12 || height < 12 {
        return Err("Input image must be at least 12x12 pixels".into());
    }
    let output_width = width / PATCH * PATCH - PATCH;
    let output_height = height / PATCH * PATCH - PATCH;
    if output_width == 0 || output_height == 0 {
        return Err("Input image must be at least 12x12 pixels".into());
    }
    let mut result = image::GrayImage::from_pixel(output_width, output_height, image::Luma([255]));
    let y_limit = output_height.saturating_sub(PATCH);
    let x_limit = output_width.saturating_sub(PATCH);
    let mut y = 0;
    while y < y_limit {
        let mut x = 0;
        while x < x_limit {
            let mut block = Vec::with_capacity(INPUTS);
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    block.push(gray.get_pixel(x + dx, y + dy)[0] as f64 / 256.0);
                }
            }
            let values = model.process(&block);
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let value = (values[(dy * PATCH + dx) as usize] * 256.0) as u8;
                    result.put_pixel(x + dx, y + dy, image::Luma([value]));
                }
            }
            x += 3;
        }
        y += 3;
    }
    let destination = output_path.map(Path::to_path_buf).unwrap_or_else(|| {
        let stem = image_path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
        image_path.with_file_name(format!("{stem}_result.png"))
    });
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    println!("Conversion time: {:.2} seconds", started.elapsed().as_secs_f64());
    image::DynamicImage::ImageLuma8(result).to_rgb8().save(&destination)?;
    println!("Saved converted image to {}", destination.display());
    Ok(())
}

fn usage() {
    eprintln!("Usage:");
    eprintln!("  contour_net train [--source PATH] [--target PATH] [--model PATH] [--steps N]");
    eprintln!("  contour_net convert IMAGE [--model PATH] [--output PATH]");
}

fn option_value(args: &[String], index: &mut usize, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    *index += 1;
    let value = args.get(*index).ok_or_else(|| format!("Missing value for {name}"))?;
    Ok(PathBuf::from(value))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = default_root();
    let default_source = root.join("bin").join("keanu.png");
    let default_target = root.join("bin").join("keanu_result.png");
    let default_model = root.join("bin").join("trained.json");
    match args.first().map(String::as_str) {
        Some("train") => {
            let (mut source, mut target, mut model, mut steps) =
                (default_source, default_target, default_model, 100_000usize);
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--source" => source = option_value(&args, &mut i, "--source")?,
                    "--target" => target = option_value(&args, &mut i, "--target")?,
                    "--model" => model = option_value(&args, &mut i, "--model")?,
                    "--steps" => {
                        i += 1;
                        steps = args.get(i).ok_or("Missing value for --steps")?.parse()?;
                    }
                    _ => return Err(format!("Unknown train option: {}", args[i]).into()),
                }
                i += 1;
            }
            train(&source, &target, &model, steps)
        }
        Some("convert") if args.len() >= 2 => {
            let image = PathBuf::from(&args[1]);
            let (mut model, mut output) = (default_model, None);
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--model" => model = option_value(&args, &mut i, "--model")?,
                    "--output" => output = Some(option_value(&args, &mut i, "--output")?),
                    _ => return Err(format!("Unknown convert option: {}", args[i]).into()),
                }
                i += 1;
            }
            convert(&image, &model, output.as_deref())
        }
        _ => {
            usage();
            Err("A train or convert command is required".into())
        }
    }
}





