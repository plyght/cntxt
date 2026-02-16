use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const LOCATIONS: &[&str] = &[
    "kitchen", "garden", "bedroom", "office", "library", "park", "store", "bathroom", "garage",
    "basement", "attic", "hallway", "patio", "workshop", "studio", "cellar",
];

const NAMES: &[&str] = &[
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank", "Iris", "Jack",
];

const OBJECTS: &[&str] = &[
    "key", "book", "phone", "letter", "map", "ring", "coin", "hat", "bag", "card",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuizType {
    EntityLocation,
    ObjectHolder,
    Contradiction,
    TemporalDelta,
    MultiHop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quiz {
    pub question_tokens: Vec<u32>,
    pub answer_idx: usize,
    pub quiz_type: QuizType,
    pub distance: usize,
    pub num_answers: usize,
}

#[derive(Debug, Clone)]
struct WorldSnapshot {
    entity_locations: HashMap<String, String>,
    entity_objects: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
struct TransferEvent {
    object: String,
    #[allow(dead_code)]
    from_entity: String,
    to_entity: String,
    chunk_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub chunks: Vec<Vec<u32>>,
    pub quizzes: Vec<(usize, Quiz)>,
}

impl TrainingExample {
    pub fn max_quiz_answers(&self) -> usize {
        self.quizzes
            .iter()
            .map(|(_, q)| q.num_answers)
            .max()
            .unwrap_or(16)
    }
}

pub struct Vocabulary {
    pub word_to_id: HashMap<String, u32>,
    pub id_to_word: HashMap<u32, String>,
    next_id: u32,
}

impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Vocabulary {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            next_id: 1,
        };
        vocab.add_word("<pad>");
        vocab.add_word("<quiz>");
        vocab.add_word("<sep>");
        for name in NAMES {
            vocab.add_word(name);
        }
        for loc in LOCATIONS {
            vocab.add_word(loc);
        }
        for obj in OBJECTS {
            vocab.add_word(obj);
        }
        let filler_words = [
            "the", "a", "an", "is", "in", "at", "to", "went", "moved", "walked", "picked", "up",
            "put", "down", "gave", "took", "from", "and", "then", "after", "that", "later", "next",
            "while", "before", "has", "had", "was", "with", "where", "what", "who", "holds",
            "carrying", "left", "dropped", "found", "lost", "happy", "sad", "alive", "not", "dead",
            "said", "told", "asked", "knows", "thinks", "believes", "saw", "still", "changed",
            "no", "yes", "since", "between", "now", "chunk", "waited", "handed", "passed",
            "remained", "stayed", "met", "behind", "for",
        ];
        for w in &filler_words {
            vocab.add_word(w);
        }
        vocab
    }

    pub fn add_word(&mut self, word: &str) -> u32 {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }
        let id = self.next_id;
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.insert(id, word.to_string());
        self.next_id += 1;
        id
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|w| {
                let lower = w.to_lowercase();
                if let Some(&id) = self.word_to_id.get(&lower) {
                    id
                } else if let Some(&id) = self.word_to_id.get(w) {
                    id
                } else {
                    self.add_word(&lower)
                }
            })
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| {
                self.id_to_word
                    .get(id)
                    .map(|s| s.as_str())
                    .unwrap_or("<unk>")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn size(&self) -> usize {
        self.next_id as usize
    }

    pub fn get_id(&self, word: &str) -> Option<u32> {
        self.word_to_id.get(word).copied()
    }
}

pub struct EntityTrackingGenerator {
    rng: ChaCha8Rng,
}

impl EntityTrackingGenerator {
    pub fn new(seed: u64) -> Self {
        EntityTrackingGenerator {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn generate(&mut self, vocab: &mut Vocabulary, num_chunks: usize) -> TrainingExample {
        let num_entities = self.rng.gen_range(3..=6).min(NAMES.len());
        let names: Vec<&str> = NAMES.iter().copied().take(num_entities).collect();

        let mut entity_locations: HashMap<String, String> = HashMap::new();
        let mut entity_objects: HashMap<String, Vec<String>> = HashMap::new();

        for name in &names {
            let loc = LOCATIONS[self.rng.gen_range(0..LOCATIONS.len())];
            entity_locations.insert(name.to_string(), loc.to_string());
            entity_objects.insert(name.to_string(), Vec::new());
        }

        let num_objects = self.rng.gen_range(2..=4).min(OBJECTS.len());
        for i in 0..num_objects {
            let holder = &names[self.rng.gen_range(0..names.len())];
            entity_objects
                .get_mut(*holder)
                .unwrap()
                .push(OBJECTS[i].to_string());
        }

        let mut snapshots: Vec<WorldSnapshot> = Vec::with_capacity(num_chunks + 1);
        let mut transfer_events: Vec<TransferEvent> = Vec::new();

        snapshots.push(WorldSnapshot {
            entity_locations: entity_locations.clone(),
            entity_objects: entity_objects.clone(),
        });

        let mut chunks = Vec::with_capacity(num_chunks);
        let mut quizzes = Vec::new();

        for chunk_idx in 0..num_chunks {
            let mut sentences = Vec::new();

            let num_events = self.rng.gen_range(2..=4);
            for _ in 0..num_events {
                let event_type = self.rng.gen_range(0..3);
                let entity_idx = self.rng.gen_range(0..names.len());
                let entity = names[entity_idx];

                let sentence = match event_type {
                    0 => {
                        let new_loc = LOCATIONS[self.rng.gen_range(0..LOCATIONS.len())];
                        entity_locations.insert(entity.to_string(), new_loc.to_string());
                        let movement_templates = [
                            format!("{} went to the {}", entity, new_loc),
                            format!("{} moved to the {}", entity, new_loc),
                            format!("{} walked to the {}", entity, new_loc),
                            format!("{} is now in the {}", entity, new_loc),
                            format!("{} left for the {}", entity, new_loc),
                        ];
                        movement_templates[self.rng.gen_range(0..movement_templates.len())].clone()
                    }
                    1 => {
                        if !entity_objects[entity].is_empty() && self.rng.gen_bool(0.5) {
                            let obj_idx = self.rng.gen_range(0..entity_objects[entity].len());
                            let obj = entity_objects.get_mut(entity).unwrap().remove(obj_idx);
                            let drop_templates = [
                                format!("{} dropped the {}", entity, obj),
                                format!("{} put down the {}", entity, obj),
                                format!("{} left the {} behind", entity, obj),
                            ];
                            drop_templates[self.rng.gen_range(0..drop_templates.len())].clone()
                        } else {
                            let available: Vec<&str> = OBJECTS
                                .iter()
                                .copied()
                                .filter(|o| {
                                    !entity_objects.values().any(|v| v.contains(&o.to_string()))
                                })
                                .collect();
                            if !available.is_empty() {
                                let obj = available[self.rng.gen_range(0..available.len())];
                                entity_objects
                                    .get_mut(entity)
                                    .unwrap()
                                    .push(obj.to_string());
                                let pickup_templates = [
                                    format!("{} picked up the {}", entity, obj),
                                    format!("{} took the {}", entity, obj),
                                    format!("{} found the {}", entity, obj),
                                ];
                                pickup_templates[self.rng.gen_range(0..pickup_templates.len())].clone()
                            } else {
                                let loc = &entity_locations[entity];
                                format!("{} is in the {}", entity, loc)
                            }
                        }
                    }
                    _ => {
                        if names.len() > 1 {
                            let mut other_idx = self.rng.gen_range(0..names.len());
                            while other_idx == entity_idx {
                                other_idx = self.rng.gen_range(0..names.len());
                            }
                            let other = names[other_idx];
                            if !entity_objects[entity].is_empty() && self.rng.gen_bool(0.3) {
                                let obj_idx = self.rng.gen_range(0..entity_objects[entity].len());
                                let obj = entity_objects.get_mut(entity).unwrap().remove(obj_idx);
                                entity_objects.get_mut(other).unwrap().push(obj.clone());
                                transfer_events.push(TransferEvent {
                                    object: obj.clone(),
                                    from_entity: entity.to_string(),
                                    to_entity: other.to_string(),
                                    chunk_idx,
                                });
                                let transfer_templates = [
                                    format!("{} gave the {} to {}", entity, obj, other),
                                    format!("{} handed the {} to {}", entity, obj, other),
                                    format!("{} passed the {} to {}", entity, obj, other),
                                ];
                                transfer_templates[self.rng.gen_range(0..transfer_templates.len())].clone()
                            } else {
                                let entity_loc = &entity_locations[entity];
                                let other_loc = &entity_locations[other];
                                let see_templates: Vec<String> = if entity_loc == other_loc {
                                    vec![
                                        format!("{} saw {} in the {}", entity, other, entity_loc),
                                        format!("{} met {} in the {}", entity, other, entity_loc),
                                        format!("{} was with {} in the {}", entity, other, entity_loc),
                                    ]
                                } else {
                                    vec![format!("{} saw {} in the {}", entity, other, other_loc)]
                                };
                                see_templates[self.rng.gen_range(0..see_templates.len())].clone()
                            }
                        } else {
                            let loc = &entity_locations[entity];
                            let wait_templates = [
                                format!("{} waited in the {}", entity, loc),
                                format!("{} stayed in the {}", entity, loc),
                                format!("{} remained in the {}", entity, loc),
                            ];
                            wait_templates[self.rng.gen_range(0..wait_templates.len())].clone()
                        }
                    }
                };
                sentences.push(sentence);
            }

            let connectors = [" then ", " after that ", " later ", " next "];
            let chunk_text = sentences
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    if i == 0 {
                        s.clone()
                    } else {
                        format!(
                            "{}{}",
                            connectors[self.rng.gen_range(0..connectors.len())],
                            s
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            let chunk_tokens = vocab.encode(&chunk_text);
            chunks.push(chunk_tokens);

            snapshots.push(WorldSnapshot {
                entity_locations: entity_locations.clone(),
                entity_objects: entity_objects.clone(),
            });
        }

        self.schedule_quizzes(
            vocab,
            &names,
            &snapshots,
            &transfer_events,
            num_chunks,
            &mut quizzes,
        );

        TrainingExample { chunks, quizzes }
    }

    fn schedule_quizzes(
        &mut self,
        vocab: &mut Vocabulary,
        names: &[&str],
        snapshots: &[WorldSnapshot],
        transfer_events: &[TransferEvent],
        num_chunks: usize,
        quizzes: &mut Vec<(usize, Quiz)>,
    ) {
        // Scale target distances with num_chunks to test longer-range recall.
        // Short sequences: 1,2,3,5,7. Longer: add 10,15,20... up to ~75% of num_chunks.
        let mut target_distances: Vec<usize> = vec![1, 2, 3, 5, 7]
            .into_iter()
            .filter(|d| *d < num_chunks)
            .collect();
        if num_chunks >= 12 {
            target_distances.extend([10].iter().filter(|d| **d < num_chunks));
        }
        if num_chunks >= 18 {
            target_distances.extend([15].iter().filter(|d| **d < num_chunks));
        }
        if num_chunks >= 25 {
            target_distances.push(20);
            let long_dist = num_chunks * 3 / 4;
            if long_dist > 20 && long_dist < num_chunks {
                target_distances.push(long_dist);
            }
        }
        target_distances.sort();
        target_distances.dedup();

        for &target_dist in &target_distances {
            let chunk_idx = if target_dist < num_chunks {
                target_dist
            } else {
                num_chunks - 1
            };
            if chunk_idx == 0 {
                continue;
            }

            let quiz_types = self.available_quiz_types(
                names,
                snapshots,
                transfer_events,
                chunk_idx,
            );

            if quiz_types.is_empty() {
                continue;
            }

            let qt_idx = self.rng.gen_range(0..quiz_types.len());
            let qt = quiz_types[qt_idx].clone();

            if let Some(quiz) = self.generate_quiz(
                vocab,
                &qt,
                names,
                snapshots,
                transfer_events,
                chunk_idx,
                target_dist,
            ) {
                quizzes.push((chunk_idx, quiz));
            }
        }

        for chunk_idx in 1..num_chunks {
            if quizzes.iter().any(|(ci, _)| *ci == chunk_idx) {
                continue;
            }
            if !self.rng.gen_bool(0.5) {
                continue;
            }

            let distance = chunk_idx;
            let qt = if self.rng.gen_bool(0.6) {
                QuizType::EntityLocation
            } else {
                QuizType::ObjectHolder
            };

            if let Some(quiz) = self.generate_quiz(
                vocab,
                &qt,
                names,
                snapshots,
                transfer_events,
                chunk_idx,
                distance,
            ) {
                quizzes.push((chunk_idx, quiz));
            }
        }
    }

    fn available_quiz_types(
        &self,
        names: &[&str],
        snapshots: &[WorldSnapshot],
        transfer_events: &[TransferEvent],
        chunk_idx: usize,
    ) -> Vec<QuizType> {
        let snapshot_idx = chunk_idx + 1;
        if snapshot_idx >= snapshots.len() {
            return vec![];
        }
        let current = &snapshots[snapshot_idx];

        let mut types = vec![QuizType::EntityLocation];

        let has_objects = current.entity_objects.values().any(|v| !v.is_empty());
        if has_objects {
            types.push(QuizType::ObjectHolder);
        }

        // Contradiction and TemporalDelta both require entity location changes.
        // Compute once to avoid duplicate logic.
        if chunk_idx >= 2 && snapshot_idx < snapshots.len() {
            let earlier_idx = if chunk_idx > 3 { chunk_idx - 3 } else { 0 };
            let earlier = &snapshots[earlier_idx];
            let current = &snapshots[snapshot_idx];
            let has_location_change = names.iter().any(|name| {
                earlier.entity_locations.get(*name) != current.entity_locations.get(*name)
            });
            if has_location_change {
                types.push(QuizType::Contradiction);
                types.push(QuizType::TemporalDelta);
            }
        }

        let relevant_transfers: Vec<&TransferEvent> = transfer_events
            .iter()
            .filter(|t| t.chunk_idx < chunk_idx)
            .collect();
        if !relevant_transfers.is_empty() {
            types.push(QuizType::MultiHop);
        }

        types
    }

    fn generate_quiz(
        &mut self,
        vocab: &mut Vocabulary,
        quiz_type: &QuizType,
        names: &[&str],
        snapshots: &[WorldSnapshot],
        transfer_events: &[TransferEvent],
        chunk_idx: usize,
        distance: usize,
    ) -> Option<Quiz> {
        // Use snapshot at chunk_idx+1: state after processing chunk chunk_idx.
        let snapshot_idx = chunk_idx + 1;
        if snapshot_idx >= snapshots.len() {
            return None;
        }
        let current = &snapshots[snapshot_idx];

        match quiz_type {
            QuizType::EntityLocation => {
                let entity = names[self.rng.gen_range(0..names.len())];
                let question = format!("where is {}", entity);
                let question_tokens = vocab.encode(&question);
                let correct_loc = current.entity_locations.get(entity)?;
                let answer_idx = LOCATIONS.iter().position(|l| l == correct_loc)?;
                Some(Quiz {
                    question_tokens,
                    answer_idx,
                    quiz_type: QuizType::EntityLocation,
                    distance,
                    num_answers: LOCATIONS.len(),
                })
            }
            QuizType::ObjectHolder => {
                let holders: Vec<&&str> = names
                    .iter()
                    .filter(|n| {
                        current.entity_objects
                            .get(**n)
                            .map(|v| !v.is_empty())
                            .unwrap_or(false)
                    })
                    .collect();
                if holders.is_empty() {
                    return None;
                }
                let holder = holders[self.rng.gen_range(0..holders.len())];
                let objs = current.entity_objects.get(*holder)?;
                let obj = &objs[self.rng.gen_range(0..objs.len())];
                let question = format!("who holds the {}", obj);
                let question_tokens = vocab.encode(&question);
                let answer_idx = names.iter().position(|n| n == holder)?;
                Some(Quiz {
                    question_tokens,
                    answer_idx,
                    quiz_type: QuizType::ObjectHolder,
                    distance,
                    num_answers: names.len(),
                })
            }
            QuizType::Contradiction => {
                if snapshot_idx >= snapshots.len() {
                    return None;
                }
                let earlier_idx = if chunk_idx > 3 { chunk_idx - 3 } else { 0 };
                let earlier = &snapshots[earlier_idx];
                let current = &snapshots[snapshot_idx];

                let moved_entities: Vec<&&str> = names
                    .iter()
                    .filter(|name| {
                        earlier.entity_locations.get(**name) != current.entity_locations.get(**name)
                    })
                    .collect();

                if moved_entities.is_empty() {
                    return None;
                }

                let entity = moved_entities[self.rng.gen_range(0..moved_entities.len())];

                let use_old_location = self.rng.gen_bool(0.5);
                if use_old_location {
                    let old_loc = earlier.entity_locations.get(*entity)?;
                    let question = format!("is {} still in the {}", entity, old_loc);
                    let question_tokens = vocab.encode(&question);
                    Some(Quiz {
                        question_tokens,
                        answer_idx: 0,
                        quiz_type: QuizType::Contradiction,
                        distance,
                        num_answers: 2,
                    })
                } else {
                    let new_loc = current.entity_locations.get(*entity)?;
                    let question = format!("is {} still in the {}", entity, new_loc);
                    let question_tokens = vocab.encode(&question);
                    Some(Quiz {
                        question_tokens,
                        answer_idx: 1,
                        quiz_type: QuizType::Contradiction,
                        distance,
                        num_answers: 2,
                    })
                }
            }
            QuizType::TemporalDelta => {
                if snapshot_idx >= snapshots.len() {
                    return None;
                }
                let earlier_idx = if chunk_idx > 3 { chunk_idx - 3 } else { 0 };
                let earlier = &snapshots[earlier_idx];
                let current = &snapshots[snapshot_idx];

                let changed_entities: Vec<(usize, &&str)> = names
                    .iter()
                    .enumerate()
                    .filter(|(_, name)| {
                        earlier.entity_locations.get(**name) != current.entity_locations.get(**name)
                    })
                    .collect();

                if changed_entities.is_empty() {
                    return None;
                }

                let (answer_idx, _entity) =
                    changed_entities[self.rng.gen_range(0..changed_entities.len())];
                let question = format!("who changed since chunk {}", earlier_idx);
                let question_tokens = vocab.encode(&question);
                Some(Quiz {
                    question_tokens,
                    answer_idx,
                    quiz_type: QuizType::TemporalDelta,
                    distance,
                    num_answers: names.len(),
                })
            }
            QuizType::MultiHop => {
                let relevant: Vec<&TransferEvent> = transfer_events
                    .iter()
                    .filter(|t| t.chunk_idx < chunk_idx)
                    .collect();

                if relevant.is_empty() {
                    return None;
                }

                let transfer = relevant[self.rng.gen_range(0..relevant.len())];

                let holder = &transfer.to_entity;
                let holder_loc = current.entity_locations.get(holder.as_str())?;

                let question = format!("where is the {}", transfer.object);
                let question_tokens = vocab.encode(&question);
                let answer_idx = LOCATIONS.iter().position(|l| *l == holder_loc.as_str())?;

                Some(Quiz {
                    question_tokens,
                    answer_idx,
                    quiz_type: QuizType::MultiHop,
                    distance,
                    num_answers: LOCATIONS.len(),
                })
            }
        }
    }

    pub fn generate_batch(
        &mut self,
        vocab: &mut Vocabulary,
        batch_size: usize,
        num_chunks: usize,
    ) -> Vec<TrainingExample> {
        (0..batch_size)
            .map(|_| self.generate(vocab, num_chunks))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary() {
        let mut vocab = Vocabulary::new();
        let tokens = vocab.encode("Alice went to the kitchen");
        assert!(!tokens.is_empty());
        let decoded = vocab.decode(&tokens);
        assert_eq!(decoded, "Alice went to the kitchen");
    }

    #[test]
    fn test_generation() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(42);
        let example = gen.generate(&mut vocab, 5);
        assert_eq!(example.chunks.len(), 5);
        assert!(!example.chunks[0].is_empty());
    }

    #[test]
    fn test_quiz_generation() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(42);
        let example = gen.generate(&mut vocab, 10);
        assert!(
            !example.quizzes.is_empty(),
            "should generate quizzes for 10 chunks"
        );
        for (chunk_idx, quiz) in &example.quizzes {
            assert!(*chunk_idx > 0);
            assert!(!quiz.question_tokens.is_empty());
            assert!(quiz.num_answers >= 2);
        }
    }

    #[test]
    fn test_quiz_type_diversity() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(123);
        let mut found_types: Vec<QuizType> = Vec::new();
        for _ in 0..50 {
            let example = gen.generate(&mut vocab, 10);
            for (_, quiz) in &example.quizzes {
                if !found_types.iter().any(|t| *t == quiz.quiz_type) {
                    found_types.push(quiz.quiz_type.clone());
                }
            }
        }
        assert!(
            found_types.len() >= 3,
            "should generate at least 3 quiz types across 50 examples, got: {:?}",
            found_types
        );
    }

    #[test]
    fn test_scheduled_distances() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(999);
        let mut all_distances: Vec<usize> = Vec::new();
        for _ in 0..20 {
            let example = gen.generate(&mut vocab, 10);
            for (_, quiz) in &example.quizzes {
                all_distances.push(quiz.distance);
            }
        }
        assert!(
            all_distances.iter().any(|d| *d >= 3),
            "should have quizzes at distance >= 3"
        );
    }

    #[test]
    fn test_contradiction_quiz() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(777);
        let mut found_contradiction = false;
        for _ in 0..100 {
            let example = gen.generate(&mut vocab, 10);
            for (_, quiz) in &example.quizzes {
                if quiz.quiz_type == QuizType::Contradiction {
                    assert_eq!(quiz.num_answers, 2);
                    assert!(quiz.answer_idx <= 1);
                    found_contradiction = true;
                }
            }
        }
        assert!(
            found_contradiction,
            "should find at least one contradiction quiz in 100 examples"
        );
    }

    #[test]
    fn test_multihop_quiz() {
        let mut vocab = Vocabulary::new();
        let mut gen = EntityTrackingGenerator::new(555);
        let mut found_multihop = false;
        for _ in 0..100 {
            let example = gen.generate(&mut vocab, 10);
            for (_, quiz) in &example.quizzes {
                if quiz.quiz_type == QuizType::MultiHop {
                    assert_eq!(quiz.num_answers, LOCATIONS.len());
                    found_multihop = true;
                }
            }
        }
        assert!(
            found_multihop,
            "should find at least one multi-hop quiz in 100 examples"
        );
    }
}
