(** Basic LSTM example. *)

open Backprop

let () =
  Printexc.record_backtrace true;
  Random.self_init ()

let () =
  let inputs = 4 in
  let hidden_size = 8 in
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
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size
  in
  let net s x = Net.Vector.RNN.long_short_term_memory ~weight_state ~weight ~bias s x in
  let s0 = Net.cst (Vector.uniform hidden_size, Vector.uniform hidden_size) in
  let x0 = Net.observe_descent (fun _ -> print_endline "Backpropagated!") @@ Net.cst @@ Vector.uniform inputs in
  let s, y = net s0 x0 in
  Net.Vector.drop_pair s;
  Net.Vector.squared_distance_to (Vector.uniform hidden_size) y |> Net.descent 0.1 |> Net.run
