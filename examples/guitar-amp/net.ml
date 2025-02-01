open Backprop
open Extlib

let inputs = 1 (* 2 for stereo *)
let hidden_size = 8

(* Create a new state. *)
let state () =
  Vector.uniform hidden_size, Vector.uniform hidden_size

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
  let net s x =
    Net.Vector.RNN.long_short_term_memory ~weight_state ~weight ~bias s x
    |> Pair.map_right Net.Vector.to_scalar
  in
  fun state y x ->
    let state = Net.cst state in
    let state, out =
      x
      |> Array.map (Array.make 1)
      |> Array.map Net.cst
      |> Net.Vector.RNN.bulk net state
    in
    let s = Net.eval state in
    let o = Net.eval out in
    (* Optimize *)
    Net.Vector.drop_pair state;
    begin
      out
      |> Net.Vector.squared_distance_to y
      |> Net.descent 0.1
      |> Net.run
    end;
    s, o
