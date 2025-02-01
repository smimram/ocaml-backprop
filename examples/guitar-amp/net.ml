open Backprop

let inputs = 1 (* 2 for stereo *)
let hidden_size = 8

let net =
  let weight_state = ref (Vector.Linear.uniform hidden_size hidden_size, Vector.Linear.uniform hidden_size hidden_size, Vector.Linear.uniform hidden_size hidden_size, Vector.Linear.uniform hidden_size hidden_size) in
  let weight = ref (Vector.Linear.uniform inputs hidden_size, Vector.Linear.uniform inputs hidden_size, Vector.Linear.uniform inputs hidden_size, Vector.Linear.uniform inputs hidden_size) in
  Backpropagatable.Vector.RNN.long_short_term_memory ~weight_state ~weight
