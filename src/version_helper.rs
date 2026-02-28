use walkdir::WalkDir;

// Get the version from the model file name
fn extract_version(file_name: &str) -> Option<u64> {
    let without_ext = file_name.strip_suffix(".bin")?;
    let (_, version_part) = without_ext.split_once('_')?;
    version_part.parse::<u64>().ok()
}

// Get the next version of the model
pub fn get_next_model_version_index(path: &str) -> u64 {
    let mut highest_version: u64 = 0;
    for entry in WalkDir::new(path).into_iter().filter_map(Result::ok) {
        let file_name = match entry.file_name().to_str() {
            Some(name) => name,
            None => continue,
        };

        if let Some(version) = extract_version(file_name) {
            if version > highest_version {
                highest_version = version;
            }
        }
    }
    highest_version + 1
}
