(** Learn the square function with a simple net. *)

open Backprop

let () = Random.self_init ()

(* Let's try to learn f(x) = x². *)
let () =
  (* Generate dataset. *)
  let dataset = List.map (fun x -> x, x*.x) [-1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0] in

  (* Train a network with one hidden layer of size 6. *)
  let n = 6 in
  let layer1 =
    let weights = ref @@ Vector.Linear.uniform 1 n in
    let bias = ref @@ Vector.zero n in
    Net.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  let layer2 =
    let weights = ref @@ Vector.Linear.uniform n 1 in
    let bias = ref @@ Vector.zero 1 in
    Net.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  let net x = x |> layer1 |> layer2 in
  for _ = 0 to 10_000 do
    (* Net.fit net dataset *)
    List.iter
      (fun (x,y) ->
         let y = Vector.scalar y in
         let descent =
           Vector.scalar x
           |> Net.cst
           |> net
           |> Net.Vector.squared_distance_to y
           |> Net.descent 0.2
         in
         Net.run descent
      ) dataset
  done;

  (* Profit. *)
  let xs = [-1.0; -0.5; 0.0; 0.1; 0.5; 1.] in
  List.iter
    (fun x ->
       let y = x |> Vector.scalar |> Net.cst |> net |> Net.eval |> Vector.to_scalar in
       Printf.printf "f(%f) = %f instead of %f\n" x y (x *. x)
    ) xs
