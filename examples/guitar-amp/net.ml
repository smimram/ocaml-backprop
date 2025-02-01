open Backprop

let inputs = 1 (* 2 for stereo *)
let hidden_size = 8

let net =
  let weight_state =
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size,
    ref @@ Vector.Linear.uniform hidden_size hidden_size
  in
  let weight =
    ref @@ Vector.Linear.uniform inputs hidden_size,
    ref @@ Vector.Linear.uniform inputs hidden_size,
    ref @@ Vector.Linear.uniform inputs hidden_size,
    ref @@ Vector.Linear.uniform inputs hidden_size
  in
  let bias =
    ref @@ Vector.uniform hidden_size,
    ref @@ Vector.uniform hidden_size,
    ref @@ Vector.uniform hidden_size,
    ref @@ Vector.uniform hidden_size
  in
  let state =
    Vector.uniform hidden_size,
    Vector.uniform hidden_size
  in
  let state = Net.cst state in
  let net = Net.Vector.RNN.long_short_term_memory ~weight_state ~weight ~bias in
  fun x ->
    x
    |> Array.map Net.cst
    |> Net.Vector.RNN.bulk net state
