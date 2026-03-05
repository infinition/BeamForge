use std::collections::{HashMap, BinaryHeap};
use std::fs::{self, File};
use std::io::{self, Write, BufWriter, BufReader};
use std::cmp::Ordering;
use rand::Rng;
use serde::{Serialize, Deserialize};
use walkdir::WalkDir;

// --- CONFIGURATION ---
const GRAVITY: f32 = 1.0;
const MAX_WEIGHT: f32 = 50.0;
const BEAM_WIDTH: usize = 5; 
const MAX_GENERATION_LENGTH: usize = 30;

// =====================================================================
// 1. TOKENIZER (For absolute performance)
// =====================================================================
#[derive(Serialize, Deserialize, Clone)]
struct Tokenizer {
    word_to_id: HashMap<String, u32>,
    id_to_word: HashMap<u32, String>,
    next_id: u32,
}

impl Tokenizer {
    fn new() -> Self {
        Tokenizer {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            next_id: 1, 
        }
    }

    fn get_or_create_id(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.word_to_id.get(word) {
            id
        } else {
            let id = self.next_id;
            self.word_to_id.insert(word.to_string(), id);
            self.id_to_word.insert(id, word.to_string());
            self.next_id += 1;
            id
        }
    }

    fn find_id(&self, word: &str) -> Option<u32> {
        self.word_to_id.get(&word.to_lowercase()).cloned()
    }

    fn decode(&self, id: u32) -> String {
        self.id_to_word.get(&id).cloned().unwrap_or_else(|| "<?>".to_string())
    }

    fn clean_text(text: &str) -> Vec<String> {
        let cleaned = text
            .replace("\r\n", "\n")
            .replace("\n\n", " . ") 
            .replace("\n", " ")
            .replace("#", "")
            .replace("*", "")
            .replace("-", "")
            .replace("`", "")
            .replace("|", "")
            .replace(".", " . ")
            .replace(",", " , ")
            .replace("!", " ! ")
            .replace("?", " ? ")
            .replace(":", " : ")
            .replace(";", " ; ")
            .replace("'", " ' ")
            .replace("’", " ' ")
            .to_lowercase();

        cleaned.split_whitespace().map(|s| s.to_string()).collect()
    }

    fn is_sentence_ender(&self, id: u32) -> bool {
        let w = self.decode(id);
        [".", "!", "?"].contains(&w.as_str())
    }

    fn is_punctuation(&self, id: u32) -> bool {
        let w = self.decode(id);
        [".", ",", "!", "?", ";", ":", "'"].contains(&w.as_str())
    }
}

// =====================================================================
// 2. BEAM NODE (Exploration scenario)
// =====================================================================
#[derive(Clone, PartialEq)]
struct BeamNode {
    score: f32,               // Additive raw score
    normalized_score: f32,    // Average score per word (THE FIX)
    tokens: Vec<u32>,         // Cloning a Vec<u32> is infinitely faster than Vec<String>
    current_w1: u32,
    current_w2: u32,
}

impl Eq for BeamNode {}
impl PartialOrd for BeamNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.normalized_score.partial_cmp(&other.normalized_score)
    }
}
impl Ord for BeamNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
struct ContextKey {
    w1: u32,
    w2: u32,
}

// =====================================================================
// 3. SEMANTIC MESH (The Brain)
// =====================================================================
#[derive(Serialize, Deserialize)]
struct SemanticMesh {
    tokenizer: Tokenizer,
    synapses: HashMap<ContextKey, HashMap<u32, f32>>,
    simple_links: HashMap<u32, HashMap<u32, f32>>,
    fast_forward: HashMap<u32, Vec<u32>>,
}

impl SemanticMesh {
    fn new() -> Self {
        SemanticMesh {
            tokenizer: Tokenizer::new(),
            synapses: HashMap::new(),
            simple_links: HashMap::new(),
            fast_forward: HashMap::new(),
        }
    }

    fn learn(&mut self, input: &str) {
        let words = Tokenizer::clean_text(input);
        if words.len() < 3 { return; }

        let mut tokens = Vec::with_capacity(words.len());
        for w in words {
            tokens.push(self.tokenizer.get_or_create_id(&w));
        }

        for i in 0..tokens.len() - 2 {
            let w1 = tokens[i];
            let w2 = tokens[i+1];
            let target = tokens[i+2];

            let key = ContextKey { w1, w2 };

            // 1. Synapses (Triangulation)
            let outputs = self.synapses.entry(key.clone()).or_insert(HashMap::new());
            let weight = outputs.entry(target).or_insert(0.0);
            *weight += GRAVITY;
            if *weight > MAX_WEIGHT { *weight = MAX_WEIGHT; }

            // 2. Simple Links (Fallback)
            let simple_outs = self.simple_links.entry(w2).or_insert(HashMap::new());
            let s_weight = simple_outs.entry(target).or_insert(0.0);
            *s_weight += GRAVITY * 0.5;
            if *s_weight > MAX_WEIGHT { *s_weight = MAX_WEIGHT; }

            // 3. Fast Forward
            let nexts = self.fast_forward.entry(w1).or_insert(Vec::new());
            if nexts.len() < 20 && !nexts.contains(&w2) {
                nexts.push(w2);
            }
        }
    }

    fn ingest_folder(&mut self, folder_path: &str) {
        println!(">>> INGESTION (Tokenized Beam Search) : {}", folder_path);
        let mut files_count = 0;
        for entry in WalkDir::new(folder_path).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "txt" || ext == "md") {
                print!("Reading {:?} ... ", entry.file_name());
                io::stdout().flush().unwrap();
                if let Ok(text) = fs::read_to_string(path) {
                    self.learn(&text);
                    println!("OK.");
                    files_count += 1;
                }
            }
        }
        println!(">>> Finished. {} files.", files_count);
    }

    // --- BEAM SEARCH ENGINE CORRECTION ---
    fn generate_beam(&self, w1_id: u32, w2_id: u32) {
        let mut beams = BinaryHeap::new();
        
        // Initial Seed
        beams.push(BeamNode {
            score: 0.0,
            normalized_score: 0.0,
            tokens: vec![w1_id, w2_id],
            current_w1: w1_id,
            current_w2: w2_id,
        });

        for _ in 0..MAX_GENERATION_LENGTH {
            let mut candidates = BinaryHeap::new();
            let mut found_endpoint = false;

            while let Some(node) = beams.pop() {
                // If sentence is finished (punctuation)
                if self.tokenizer.is_sentence_ender(node.current_w2) {
                    let mut final_node = node.clone();
                    final_node.score += 15.0; // Small bonus for finishing
                    // LENGTH PENALTY : divide by length to avoid infinite sentences
                    final_node.normalized_score = final_node.score / (final_node.tokens.len() as f32);
                    candidates.push(final_node);
                    found_endpoint = true;
                    continue;
                }

                let key = ContextKey { w1: node.current_w1, w2: node.current_w2 };
                
                // Search for next words (Synapses or Fallbacks)
                let next_map_opt = self.synapses.get(&key).or_else(|| self.simple_links.get(&node.current_w2));

                if let Some(next_map) = next_map_opt {
                    for (&target_id, &weight) in next_map {
                        let mut new_tokens = node.tokens.clone();
                        new_tokens.push(target_id);
                        
                        let random_jitter: f32 = rand::thread_rng().gen_range(0.9..1.1);
                        let new_score = node.score + (weight * random_jitter);
                        let new_normalized = new_score / (new_tokens.len() as f32); // LENGTH PENALTY

                        candidates.push(BeamNode {
                            score: new_score,
                            normalized_score: new_normalized,
                            tokens: new_tokens,
                            current_w1: node.current_w2,
                            current_w2: target_id,
                        });
                    }
                } else {
                    candidates.push(node); // Dead end
                }
            }

            // PRUNING: Keep only the best scenarios (BEAM_WIDTH)
            beams.clear();
            for _ in 0..BEAM_WIDTH {
                if let Some(best) = candidates.pop() {
                    beams.push(best);
                }
            }
            
            // If all best scenarios finished their sentences, stop
            if found_endpoint && beams.iter().all(|n| self.tokenizer.is_sentence_ender(n.current_w2)) {
                break;
            }
        }

        // Print the best final scenario
        if let Some(best_beam) = beams.pop() {
            print!("\nBEAMFORGE:");
            for &tid in &best_beam.tokens {
                let w = self.tokenizer.decode(tid);
                if self.tokenizer.is_punctuation(tid) {
                    print!("{}", w);
                } else {
                    print!(" {}", w);
                }
            }
            println!("\n");
        } else {
            println!("\nBEAMFORGE: (...)\n");
        }
    }

    fn generate(&self, prompt: &str) {
        let clean_prompt = prompt.replace("?", ""); 
        let clean_input = Tokenizer::clean_text(&clean_prompt);
        if clean_input.is_empty() { return; }

        let mut known_ids = Vec::new();
        for w in clean_input {
            if let Some(id) = self.tokenizer.find_id(&w) {
                known_ids.push(id);
            }
        }

        if known_ids.is_empty() {
            println!("(?) I don't know these words.");
            return;
        }

        let (w1, w2) = if known_ids.len() == 1 {
            let start_id = known_ids[0];
            if let Some(nexts) = self.fast_forward.get(&start_id) {
                let mut rng = rand::thread_rng();
                let next_id = nexts[rng.gen_range(0..nexts.len())];
                (start_id, next_id)
            } else {
                if let Some(simples) = self.simple_links.get(&start_id) {
                     let total: f32 = simples.values().sum();
                     let mut pick = rand::thread_rng().gen_range(0.0..total);
                     let mut res_id = *simples.keys().next().unwrap();
                     for (&id, &s) in simples {
                         pick -= s;
                         if pick <= 0.0 { res_id = id; break; }
                     }
                     (start_id, res_id)
                } else {
                    println!("(?) I lack context.");
                    return;
                }
            }
        } else {
            (known_ids[known_ids.len()-2], known_ids[known_ids.len()-1])
        };
        
        self.generate_beam(w1, w2);
    }

    fn save(&self, filename: &str) {
        println!("Saving...");
        let f = BufWriter::new(File::create(filename).unwrap());
        bincode::serialize_into(f, self).unwrap();
        println!("Ok.");
    }

    fn load(filename: &str) -> Option<Self> {
        println!("Loading...");
        if let Ok(f) = File::open(filename) {
            let reader = BufReader::new(f);
            if let Ok(mesh) = bincode::deserialize_from(reader) {
                println!("Ok.");
                return Some(mesh);
            }
        }
        println!("File not found.");
        None
    }
}

fn main() {
    println!("--- BEAMFORGE (TOKENIZED BEAM SEARCH COGNITIVE ENGINE) ---");
    println!("Commands : /train [folder], /save, /load, /quit");

    let mut mesh = SemanticMesh::new();
    if let Some(loaded) = SemanticMesh::load("beamforge.brain") { mesh = loaded; }

    loop {
        print!("YOU > ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Error");
        let prompt = input.trim();

        if prompt == "/quit" { break; }
        if prompt == "/save" { mesh.save("beamforge.brain"); continue; }
        if prompt == "/load" { if let Some(l) = SemanticMesh::load("beamforge.brain") { mesh = l; } continue; }
        
        if prompt.starts_with("/train ") {
            let folder = prompt.strip_prefix("/train ").unwrap();
            mesh.ingest_folder(folder);
        } else if !prompt.is_empty() {
            let is_question = prompt.starts_with("?") || prompt.ends_with("?");
            if is_question {
                mesh.generate(prompt);
            } else {
                mesh.learn(prompt);
                println!("[Integrated]");
            }
        }
    }
}