open Backprop
open Extlib

let inputs = 1 (* 2 for stereo *)

(* Create a new state. *)
let state hidden_size =
  Vector.uniform hidden_size, Vector.uniform hidden_size

let net hidden_size =
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
  (* Fully connected layer. *)
  let fc =
    let w = ref @@ Vector.Linear.uniform hidden_size 1 in
    let b = ref @@ Vector.uniform 1 in
    fun x ->
      Net.Vector.to_scalar @@ Net.Vector.add (Net.Vector.var b) (Net.Linear.app (Net.Linear.var w) x)
  in
  (* The network. *)
  let net s x =
    Net.Vector.RNN.long_short_term_memory ~weight_state ~weight ~bias s x
    |> Pair.map_right fc
  in
  fun ~optimize ~state y x ->
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
    if optimize then
      (
        Net.Vector.drop_pair state;
        out
        |> Net.Vector.squared_distance_to y
        |> Net.descent 0.01
        |> Net.run
      );
    s, o
