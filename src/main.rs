use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::exit;
use std::sync::Arc;

use arrow::array::{Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDateTime, Utc};
use clap::{App, Arg};
use jsonschema::{Draft, JSONSchema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{WriterProperties, WriterVersion};
use serde::{Deserialize, Serialize, de};
use serde_json::{json, Map, Value, Deserializer};
use log::{debug, error, warn};

// Initialize env_logger to target stderr.
use env_logger;

// Custom format validator for date-time using chrono
fn validate_datetime(value: &str) -> bool {
    // Accept empty strings
    if value.is_empty() {
        return true;
    }

    // Try parsing with timezone first (most common format)
    if DateTime::parse_from_rfc3339(value).is_ok() {
        return true;
    }

    // Try parsing without timezone
    if NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").is_ok() {
        return true;
    }

    // Try parsing with timezone in alternative format
    if DateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f%z").is_ok() {
        return true;
    }

    // Try parsing date only format (YYYY-MM-DD)
    if NaiveDateTime::parse_from_str(&format!("{}T00:00:00", value), "%Y-%m-%dT%H:%M:%S").is_ok() {
        return true;
    }

    false
}

fn parse_json_with_infinity(input: &str) -> Result<Value, serde_json::Error> {
    // Replace Infinity with a string representation that serde_json can handle
    let modified_input = input.replace("Infinity", "\"Infinity\"")
        .replace("-Infinity", "\"-Infinity\"")
        .replace("NaN", "\"NaN\"");
    
    // Parse the modified JSON
    let mut deserializer = Deserializer::from_str(&modified_input);
    Value::deserialize(&mut deserializer)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Config {
    destination_path: Option<String>,
    validate: Option<bool>,
    batch_size: Option<usize>,
    metrics_write_threshold: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SchemaMessage {
    #[serde(rename = "type")]
    message_type: String,
    stream: String,
    schema: Value,
    key_properties: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct RecordMessage {
    #[serde(rename = "type")]
    message_type: String,
    stream: String,
    #[serde(deserialize_with = "deserialize_record")]
    record: Value,
}

#[derive(Serialize, Deserialize, Debug)]
struct StateMessage {
    #[serde(rename = "type")]
    message_type: String,
    value: Value,
}

fn deserialize_record<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: de::Deserializer<'de>,
{
    let v = Value::deserialize(deserializer)?;
    if let Value::Object(obj) = v {
        let mut new_obj = Map::new();
        for (key, value) in obj {
            let new_value = match value {
                Value::String(s) => {
                    match s.as_str() {
                        "Infinity" | "inf" => json!(f64::INFINITY),
                        "-Infinity" | "-inf" => json!(f64::NEG_INFINITY),
                        "NaN" | "nan" => json!(f64::NAN),
                        _ => Value::String(s),
                    }
                },
                Value::Number(n) => {
                    if n.is_f64() {
                        let f = n.as_f64().unwrap();
                        if f.is_infinite() {
                            if f.is_sign_positive() {
                                json!(f64::INFINITY)
                            } else {
                                json!(f64::NEG_INFINITY)
                            }
                        } else if f.is_nan() {
                            json!(f64::NAN)
                        } else {
                            Value::Number(n)
                        }
                    } else {
                        Value::Number(n)
                    }
                },
                _ => value,
            };
            new_obj.insert(key, new_value);
        }
        Ok(Value::Object(new_obj))
    } else {
        Ok(v)
    }
}

fn emit_state(state: &Option<Value>) {
    let state_value = state.clone().unwrap_or(json!({}));
    let line = serde_json::to_string(&state_value).unwrap();
    debug!("Emitting state {}", line);
    println!("{}", line);
    io::stdout().flush().unwrap();
}

struct ParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<ArrowSchema>,
    batch_size: usize,
    current_batch: Vec<Vec<String>>,
    record_count: u64,
}

impl ParquetWriter {
    fn new(file: File, schema: ArrowSchema, batch_size: usize) -> Self {
        let props = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .build();

        ParquetWriter {
            writer: ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props)).unwrap(),
            schema: Arc::new(schema),
            batch_size,
            current_batch: Vec::new(),
            record_count: 0,
        }
    }

    fn add_record(&mut self, record: Vec<String>) -> Result<(), String> {
        self.current_batch.push(record);
        self.record_count += 1;

        if self.current_batch.len() >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    fn flush_batch(&mut self) -> Result<(), String> {
        if self.current_batch.is_empty() {
            return Ok(());
        }

        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();

        for i in 0..self.schema.fields().len() {
            let field = self.schema.field(i);
            let values: Vec<String> = self.current_batch.iter()
                .map(|row| row[i].clone())
                .collect();

            let array: Arc<dyn Array> = match field.data_type() {
                DataType::Int64 => {
                    let int_values: Vec<Option<i64>> = values.iter()
                        .map(|v| v.parse::<i64>().ok())
                        .collect();
                    Arc::new(arrow::array::Int64Array::from(int_values))
                },
                DataType::Utf8 => {
                    Arc::new(StringArray::from(values))
                },
                _ => {
                    Arc::new(StringArray::from(values))
                }
            };
            arrays.push(array);
        }

        let batch = RecordBatch::try_new(self.schema.clone(), arrays)
            .map_err(|e| e.to_string())?;

        self.writer.write(&batch)
            .map_err(|e| e.to_string())?;

        self.current_batch.clear();
        Ok(())
    }

    fn close(mut self) -> Result<(), String> {
        self.flush_batch()?;
        self.writer.close()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn write_job_metrics(dest_path: &Path, record_counts: &HashMap<String, u64>) -> Result<(), String> {
    let metrics_path = dest_path.join("job_metrics.json");
    let mut metrics = HashMap::new();
    let mut record_count_value = Map::new();
    for (stream, count) in record_counts {
        record_count_value.insert(stream.clone(), json!(count));
    }
    metrics.insert("recordCount".to_string(), Value::Object(record_count_value));
    
    match File::create(&metrics_path) {
        Ok(file) => {
            serde_json::to_writer(file, &metrics)
                .map_err(|e| format!("Failed to write job metrics: {}", e))
        },
        Err(e) => Err(format!("Failed to create job metrics file: {}", e))
    }
}

fn persist_messages(
    messages: impl BufRead,
    destination_path: &str,
    validate: bool,
    batch_size: usize,
    metrics_write_threshold: usize,
) -> Option<Value> {
    let mut state = None;
    let mut schemas = HashMap::new();
    let mut validators: HashMap<String, JSONSchema> = HashMap::new();
    let mut writers: HashMap<String, ParquetWriter> = HashMap::new();
    let mut record_counts: HashMap<String, u64> = HashMap::new();
    let mut total_records_processed = 0;

    let now = Utc::now().format("%Y%m%dT%H%M%S").to_string();
    let dest_path = Path::new(destination_path);

    // Create destination directory if it doesn't exist
    if !dest_path.exists() {
        if let Err(e) = std::fs::create_dir_all(dest_path) {
            error!("Failed to create destination directory {}: {}", dest_path.display(), e);
            return None;
        }
    }

    for line in messages.lines() {
        let message = match line {
            Ok(msg) => msg,
            Err(e) => {
                error!("Error reading line: {}", e);
                continue;
            }
        };

        let message_value: Value = match parse_json_with_infinity(&message) {
            Ok(v) => v,
            Err(e) => {
                error!("Unable to parse: {}\nError: {}", message, e);
                continue;
            }
        };

        let message_obj = match message_value.as_object() {
            Some(obj) => obj,
            None => {
                error!("Message is not a valid JSON object");
                continue;
            }
        };

        let message_type = match message_obj.get("type") {
            Some(Value::String(t)) => t,
            _ => {
                error!("Message has no type field");
                continue;
            }
        };

        match message_type.as_str() {
            "RECORD" => {
                let record_message: RecordMessage = match serde_json::from_value(message_value) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Failed to parse RECORD message: {}", e);
                        continue;
                    }
                };

                let stream = record_message.stream.replace("/", "_");

                if !schemas.contains_key(&stream) {
                    error!(
                        "A record for stream {} was encountered before a corresponding schema",
                        stream
                    );
                    continue;
                }

                if validate {
                    if let Some(validator) = validators.get(&stream) {
                        if let Err(errors) = validator.validate(&record_message.record) {
                            for error in errors {
                                error!("Validation error: {}", error);
                            }
                            panic!("Record validation failed");
                        }
                    }
                }

                if !writers.contains_key(&stream) {
                    let filename = format!("{}-{}.parquet", stream, now);
                    let file_path = dest_path.join(&filename);

                    let file = match OpenOptions::new()
                        .write(true)
                        .create(true)
                        .append(true)
                        .open(&file_path)
                    {
                        Ok(f) => f,
                        Err(e) => {
                            error!("Failed to open file {}: {}", file_path.display(), e);
                            continue;
                        }
                    };

                    let arrow_schema: &Value = schemas.get(&stream).unwrap();
                    let properties = arrow_schema.get("properties").unwrap().as_object().unwrap();
                    let fields: Vec<Field> = properties
                        .iter()
                        .map(|(name, field_schema)| {
                            let data_type = match field_schema.get("type") {
                                Some(Value::String(t)) => match t.as_str() {
                                    "integer" => DataType::Int64,
                                    "string" => DataType::Utf8,
                                    "object" => DataType::Utf8,
                                    "array" => DataType::Utf8,
                                    _ => DataType::Utf8,
                                },
                                Some(Value::Array(types)) => {
                                    // Handle union types (e.g., ["object", "string", "null"])
                                    if types.iter().any(|t| t == "null") {
                                        DataType::Utf8
                                    } else {
                                        match types.first().and_then(|t| t.as_str()) {
                                            Some("integer") => DataType::Int64,
                                            Some("string") => DataType::Utf8,
                                            Some("object") => DataType::Utf8,
                                            Some("array") => DataType::Utf8,
                                            _ => DataType::Utf8,
                                        }
                                    }
                                }
                                _ => DataType::Utf8,
                            };
                            Field::new(name, data_type, true)
                        })
                        .collect();

                    let schema = ArrowSchema::new(fields.clone());
                    writers.insert(stream.clone(), ParquetWriter::new(file, schema, batch_size));
                }

                if let Some(writer) = writers.get_mut(&stream) {
                    if let Value::Object(ref obj) = record_message.record {
                        let mut row = Vec::new();
                        for field in writer.schema.fields() {
                            let value_str = match obj.get(field.name()) {
                                Some(Value::String(s)) => s.clone(),
                                Some(Value::Number(n)) => {
                                    if n.is_f64() {
                                        let f = n.as_f64().unwrap();
                                        if f.is_infinite() {
                                            if f.is_sign_positive() {
                                                "Infinity".to_string()
                                            } else {
                                                "-Infinity".to_string()
                                            }
                                        } else if f.is_nan() {
                                            "NaN".to_string()
                                        } else {
                                            n.to_string()
                                        }
                                    } else {
                                        n.to_string()
                                    }
                                },
                                Some(Value::Bool(b)) => b.to_string(),
                                Some(Value::Object(o)) => serde_json::to_string(o).unwrap_or_default(),
                                Some(Value::Array(a)) => serde_json::to_string(a).unwrap_or_default(),
                                Some(Value::Null) | None => String::new(),
                            };
                            row.push(value_str);
                        }

                        if let Err(e) = writer.add_record(row) {
                            error!("Failed to write record: {}", e);
                            continue;
                        }
                    } else {
                        error!("Record is not a valid JSON object");
                        continue;
                    }

                    *record_counts.entry(stream.clone()).or_insert(0) += 1;
                    total_records_processed += 1;

                    // Write job metrics after threshold number of records
                    if total_records_processed % metrics_write_threshold == 0 {
                        if let Err(e) = write_job_metrics(dest_path, &record_counts) {
                            error!("{}", e);
                        }
                    }
                }
            }
            "STATE" => {
                let state_message: StateMessage = match serde_json::from_value(message_value) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Failed to parse STATE message: {}", e);
                        continue;
                    }
                };
                state = Some(state_message.value);
            }
            "SCHEMA" => {
                let schema_message: SchemaMessage = match serde_json::from_value(message_value) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Failed to parse SCHEMA message: {}", e);
                        continue;
                    }
                };

                let stream = schema_message.stream.replace("/", "_");
                schemas.insert(stream.clone(), schema_message.schema.clone());

                if validate {
                    let compiled = match JSONSchema::options()
                        .with_draft(Draft::Draft4)
                        .with_format("date-time", validate_datetime)
                        .compile(&schema_message.schema)
                    {
                        Ok(schema) => schema,
                        Err(e) => {
                            error!("Failed to compile schema for stream {}: {}", stream, e);
                            continue;
                        }
                    };
                    validators.insert(stream, compiled);
                }
            }
            _ => {
                warn!(
                    "Unknown message type {} in message {}",
                    message_type, message
                );
            }
        }
    }

    // Close all writers
    for (stream, writer) in writers {
        if let Err(e) = writer.close() {
            error!("Failed to close writer for stream {}: {}", stream, e);
        }
    }

    // Write final job metrics
    if let Err(e) = write_job_metrics(dest_path, &record_counts) {
        error!("{}", e);
    }

    state
}

fn main() {
    // Initialize env_logger so that logs are sent to stderr (terminal).
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .target(env_logger::Target::Stderr)
        .init();

    let matches = App::new("target-parquet")
        .version(env!("CARGO_PKG_VERSION", "0.1.0"))
        .about("Singer target that writes to Parquet files")
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .value_name("FILE")
                .help("Config file")
                .takes_value(true),
        )
        .get_matches();

    let config_path = matches.value_of("config");
    let config: Config = if let Some(config_file) = config_path {
        match File::open(config_file) {
            Ok(file) => match serde_json::from_reader(file) {
                Ok(config) => config,
                Err(e) => {
                    error!("Failed to parse config file: {}", e);
                    exit(1);
                }
            },
            Err(e) => {
                error!("Failed to open config file: {}", e);
                exit(1);
            }
        }
    } else {
        Config {
            destination_path: None,
            validate: None,
            batch_size: None,
            metrics_write_threshold: None,
        }
    };

    let stdin = io::stdin();
    let input_messages = BufReader::new(stdin.lock());

    let destination_path = config.destination_path.as_deref().unwrap_or("");
    let validate = config.validate.unwrap_or(true);
    let batch_size = config.batch_size.unwrap_or(1000);
    let metrics_write_threshold = config.metrics_write_threshold.unwrap_or(100);

    let state = persist_messages(
        input_messages,
        destination_path,
        validate,
        batch_size,
        metrics_write_threshold,
    );

    emit_state(&state);
    debug!("Exiting normally");
}
