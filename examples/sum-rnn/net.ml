open Backprop
open Extlib

let inputs = 1 (* 2 for stereo *)

(* Create a new state. *)
let state hidden_size =
  Vector.uniform hidden_size, Vector.uniform hidden_size

let net hidden_size vocabulary_size output_size =
  let weight_state =
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size
  in
  let weight =
    ref @@ Vector.Linear.uniform vocabulary_size hidden_size,
    ref @@ Vector.Linear.uniform vocabulary_size hidden_size,
    ref @@ Vector.Linear.uniform vocabulary_size hidden_size,
    ref @@ Vector.Linear.uniform vocabulary_size hidden_size
  in
  let bias =
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size
  in
  (* The network. *)
  let layer1 =
    Net.Vector.RNN.long_short_term_memory ~weight_state ~weight ~bias
    |> Net.Vector.RNN.bulk_state
  in
  let weight_state =
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size
  in
  let weight =
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size
  in
  let bias =
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size
  in
  let layer2 =
    Net.Vector.RNN.long_short_term_memory ~weight_state ~weight ~bias
    |> Net.Vector.RNN.bulk
  in
  let weights = ref @@ Vector.Linear.uniform hidden_size vocabulary_size in
  let bias = ref @@ Vector.zero vocabulary_size in
  let layer3 = Net.Vector.neural_network ~activation:`None ~weights:weights ~bias:bias in
  fun (*~rate*) ~state1 ~state2 (*y*) x ->
    let state1 = Net.cst state1 in
    let state2 = Net.cst state2 in
    x
    |> layer1 state1
    |> Net.unpair
    |> Pair.map_left Net.Vector.drop
    |> snd
    |> Net.Vector.repeat output_size
    |> layer2 state2
    |> Pair.map_left Net.Vector.drop_pair
    |> snd
    |> Net.Vector.demux
    |> Array.map layer3
    |> Array.map Net.softmax
    (* let s = Net.eval state in
    let o = Net.eval out in
    (* Optimize *)
    if optimize then
      (
        Net.Vector.drop_pair state;
        out
        |> Net.Vector.squared_distance_to y
        |> Net.descent rate
        |> Net.run
      );
    s, o *)
