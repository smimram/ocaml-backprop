open Backprop

let () = Random.self_init ()

(* Let's try to learn f(x) = x². *)
let () =
  (* Generate dataset. *)
  let input = [-1.0; -0.8; -0.6; -0.4; -0.2; 0.0; 0.2; 0.4; 0.6; 0.8; 1.0] in
  let datainput = List.map (fun x -> x |> Vector.scalar |> Backpropagatable.cst) input in
  let dataoutput = List.map (fun x -> x*.x |> Vector.scalar) input in
  let size_dataset = 11 in
  let size_batch = 5 in

  (* Train a network with one hidden layer of size 6. *)
  let n = 6 in  
  let layer1 =
    let weights = ref @@ Vector.Linear.uniform 1 n in
    let bias = ref @@ Vector.zero n in
    Backpropagatable.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  let layer2 =
    let weights = ref @@ Vector.Linear.uniform n 1 in
    let bias = ref @@ Vector.zero 1 in
    Backpropagatable.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  let net x = x |> layer1 |> layer2 in
  for _ = 0 to 2_000 do
    (* Net.fit net dataset *)
    let rand_data = List.init size_batch (fun _ -> Random.int size_dataset) in
    let batching l = List.map (fun n -> List.nth l n) rand_data in
    let input_batch =  batching datainput in
    let output_batch = batching dataoutput in 
    let eta = List.init size_batch (fun _ -> -0.2) in
    let descent = 
      input_batch
      |> Backpropagatable.batch net
      |> List.map2 (Backpropagatable.Vector.error_fun `Euclidean) output_batch
      |> Backpropagatable.mux
      |> Backpropagatable.climb eta
    in
    Backpropagatable.run descent
  done;

  (* Profit. *)
  let xs = [-1.0; -0.5; 0.0; 0.1; 0.5; 1.] in
  List.iter
    (fun x ->
       let y = x |> Vector.scalar |> Backpropagatable.cst |> net |> Backpropagatable.eval |> Vector.to_scalar in
       Printf.printf "f(%f) = %f instead of %f\n" x y (x *. x)
    ) xs
