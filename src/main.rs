use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::exit;
use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, NaiveDateTime, Utc};
use clap::{App, Arg};
use jsonschema::{Draft, JSONSchema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{WriterProperties, WriterVersion};
use serde::{Deserialize, Serialize, de};
use serde_json::{json, Map, Value, Deserializer};
use log::{debug, error, warn};
use regex::Regex;

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
    // Regex to match bare Infinity, -Infinity, and NaN not within quotes
    let re = Regex::new(r#"(?P<pre>[:\[\{,\s])(?P<num>-?Infinity|NaN)(?P<post>[,\}\]\s])"#).unwrap();

    let modified_input = re.replace_all(input, "${pre}\"${num}\"${post}");

    let mut deserializer = Deserializer::from_str(&modified_input);
    Value::deserialize(&mut deserializer)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Config {
    destination_path: Option<String>,
    validate: Option<bool>,
    strict_validation: Option<bool>,
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
    current_batch: Vec<Vec<Value>>,
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

    fn add_record(&mut self, record: Vec<Value>) -> Result<(), String> {
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
        let num_fields = self.schema.fields().len();
    
        for col_idx in 0..num_fields {
            let field = self.schema.field(col_idx);
            let data_type = field.data_type();
            let column_values: Vec<&Value> = self
                .current_batch
                .iter()
                .map(|row| &row[col_idx])
                .collect();
    
            let array: Arc<dyn Array> = match data_type {
                DataType::Int64 => {
                    let parsed: Vec<Option<i64>> = column_values
                        .iter()
                        .map(|v| v.as_i64())
                        .collect();
                    Arc::new(arrow::array::Int64Array::from(parsed))
                },
                DataType::Float64 => {
                    let parsed: Vec<Option<f64>> = column_values
                        .iter()
                        .map(|v| match v {
                            Value::Number(n) => n.as_f64(),
                            Value::String(s) => s.parse::<f64>().ok(), // handle Infinity/NaN
                            _ => None,
                        })
                        .collect();
                    Arc::new(arrow::array::Float64Array::from(parsed))
                },
                DataType::Boolean => {
                    let parsed: Vec<Option<bool>> = column_values
                        .iter()
                        .map(|v| v.as_bool())
                        .collect();
                    Arc::new(arrow::array::BooleanArray::from(parsed))
                },
                DataType::Utf8 => {
                    let parsed: Vec<Option<String>> = column_values
                        .iter()
                        .map(|v| match v {
                            Value::String(s) => Some(s.clone()),
                            Value::Number(n) => Some(n.to_string()),
                            Value::Bool(b) => Some(b.to_string()),
                            Value::Null => None,
                            _ => serde_json::to_string(v).ok(),
                        })
                        .collect();
                    Arc::new(arrow::array::StringArray::from(parsed))
                },
                DataType::Timestamp(_, _) => {
                    let parsed: Vec<Option<i64>> = column_values
                        .iter()
                        .map(|v| match v {
                            Value::String(s) => {
                                if s.is_empty() {
                                    return None;
                                }
                                // Try to parse ISO 8601 timestamp to milliseconds since epoch
                                if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                                    Some(dt.timestamp_millis())
                                } else if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
                                    Some(dt.and_utc().timestamp_millis())
                                } else if let Ok(dt) = DateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f%z") {
                                    Some(dt.timestamp_millis())
                                } else if let Ok(dt) = NaiveDateTime::parse_from_str(&format!("{}T00:00:00", s), "%Y-%m-%dT%H:%M:%S") {
                                    Some(dt.and_utc().timestamp_millis())
                                } else {
                                    panic!("Invalid timestamp format: {}", s);
                                }
                            },
                            Value::Null => None,
                            _ => None, // Non-string values for timestamp fields
                        })
                        .collect();
                    Arc::new(arrow::array::TimestampMillisecondArray::from(parsed))
                },
                _ => {
                    // Fallback: serialize anything as string
                    let parsed: Vec<Option<String>> = column_values
                        .iter()
                        .map(|v| serde_json::to_string(v).ok())
                        .collect();
                    Arc::new(arrow::array::StringArray::from(parsed))
                }
            };
    
            arrays.push(array);
        }
    
        let batch = RecordBatch::try_new(self.schema.clone(), arrays)
            .map_err(|e| e.to_string())?;
    
        self.writer.write(&batch).map_err(|e| e.to_string())?;
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
    strict_validation: bool,
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
                panic!("Error reading line: {}", e);
            }
        };

        let message_value: Value = match parse_json_with_infinity(&message) {
            Ok(v) => v,
            Err(e) => {
                panic!("Unable to parse: {}\nError: {}", message, e);
            }
        };

        let message_obj = match message_value.as_object() {
            Some(obj) => obj,
            None => {
                panic!("Message is not a valid JSON object");
            }
        };

        let message_type = match message_obj.get("type") {
            Some(Value::String(t)) => t,
            _ => {
                panic!("Message has no type field");
            }
        };

        match message_type.as_str() {
            "RECORD" => {
                let record_message: RecordMessage = match serde_json::from_value(message_value) {
                    Ok(m) => m,
                    Err(e) => {
                        panic!("Failed to parse RECORD message: {}", e);
                    }
                };

                let stream = record_message.stream.replace("/", "_");

                if !schemas.contains_key(&stream) {
                    panic!(
                        "A record for stream {} was encountered before a corresponding schema",
                        stream
                    );
                }

                if validate {
                    if let Some(validator) = validators.get(&stream) {
                        if let Err(errors) = validator.validate(&record_message.record) {
                            for error in errors {
                                error!("Validation error: {}", error);
                            }
                            if strict_validation {
                                panic!("Record validation failed");
                            }
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
                            panic!("Failed to open file {}: {}", file_path.display(), e);
                        }
                    };

                    let arrow_schema: &Value = schemas.get(&stream).unwrap();
                    let properties = arrow_schema.get("properties").unwrap().as_object().unwrap();
                    let fields: Vec<Field> = properties
                        .iter()
                        .map(|(name, field_schema)| {
                            let data_format = field_schema.get("format").unwrap_or(&Value::Null);

                            let data_type_str: &str = match field_schema.get("type") {
                                Some(Value::String(t)) => t.as_str(),
                                Some(Value::Array(types)) => {
                                    let type_strs: Vec<_> = types.iter()
                                        .filter_map(|t| t.as_str())
                                        .filter(|&t| t != "null")
                                        .collect();
                                    
                                    if type_strs.is_empty() {
                                        "string"
                                    } else {
                                        type_strs[0]
                                    }
                                },
                                _ => "string",
                            };

                            let data_type = match data_type_str {
                                "integer" => DataType::Int64,
                                "number" => DataType::Float64,
                                "string" => match data_format {
                                    Value::String(f) => match f.as_str() {
                                        "date-time" => DataType::Timestamp(TimeUnit::Millisecond, None),
                                        "date" => DataType::Timestamp(TimeUnit::Millisecond, None),
                                        _ => DataType::Utf8,
                                    },
                                    _ => DataType::Utf8,
                                },
                                "boolean" => DataType::Boolean,
                                "object" => DataType::Utf8,
                                "array" => DataType::Utf8,
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
                            let value = obj.get(field.name()).cloned().unwrap_or(Value::Null);
                            row.push(value);
                        }

                        if let Err(e) = writer.add_record(row) {
                            panic!("Failed to write record: {}", e);
                        }
                    } else {
                        panic!("Record is not a valid JSON object");
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
                        panic!("Failed to parse STATE message: {}", e);
                    }
                };
                state = Some(state_message.value);
            }
            "SCHEMA" => {
                let schema_message: SchemaMessage = match serde_json::from_value(message_value) {
                    Ok(m) => m,
                    Err(e) => {
                        panic!("Failed to parse SCHEMA message: {}", e);
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
                            panic!("Failed to compile schema for stream {}: {}", stream, e);
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
            strict_validation: None,
            batch_size: None,
            metrics_write_threshold: None,
        }
    };

    let stdin = io::stdin();
    let input_messages = BufReader::new(stdin.lock());

    let destination_path = config.destination_path.as_deref().unwrap_or("");
    let validate = config.validate.unwrap_or(true);
    let strict_validation = config.strict_validation.unwrap_or(false);
    let batch_size = config.batch_size.unwrap_or(1000);
    let metrics_write_threshold = config.metrics_write_threshold.unwrap_or(100);

    let state = persist_messages(
        input_messages,
        destination_path,
        validate,
        strict_validation,
        batch_size,
        metrics_write_threshold,
    );

    emit_state(&state);
    debug!("Exiting normally");
}
