(** Basic GRU example. *)

open Backprop
open Algebra

let () =
  Printexc.record_backtrace true;
  Random.self_init ()

let () =
  let inputs = 4 in
  let hidden_size = 8 in
  let weight_state =
    ref @@ Linear.uniform hidden_size hidden_size,
    ref @@ Linear.uniform hidden_size hidden_size,
    ref @@ Linear.uniform hidden_size hidden_size
  in
  let weight =
    ref @@ Linear.uniform inputs hidden_size,
    ref @@ Linear.uniform inputs hidden_size,
    ref @@ Linear.uniform inputs hidden_size
  in
  let bias =
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size,
    ref @@ Vector.zero hidden_size
  in
  let net s x = Net.Vector.RNN.gated_recurrent_unit ~weight_state ~weight ~bias s x in
  let backpropagated = ref false in
  let s0 = Net.cst @@ Vector.uniform hidden_size in
  let x0 = Net.observe_descent (fun _ -> print_endline "Backpropagated!"; backpropagated := true) @@ Net.cst @@ Vector.uniform inputs in
  let s, y = net s0 x0 in
  Net.Vector.drop s;
  Net.Vector.squared_distance_to (Vector.uniform hidden_size) y |> Net.descent 0.1 |> Net.run;
  assert !backpropagated
