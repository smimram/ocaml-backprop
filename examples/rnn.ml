open Backprop
open Backpropagatable.Vector

let () = Random.self_init ()

let () =
  let m = 5 in
  let expected = List.init m (fun n -> Vector.scalar (1.0 /. (float (m-n))**2.)) in
  let input = List.init m (fun _ -> Backpropagatable.cst [|1.0|]) in
  (* let error = List.init m (fun _ -> -0.01) in *)
  let n = 8 in
  (* Elmann *)
  (* let activations = `Relu,`None in
  let weight = ref @@ Vector.Linear.uniform 1 n in
  let weight_hidden = ref @@ Vector.Linear.uniform n n in
  let weight_output = ref @@ Vector.Linear.uniform n 1 in
  let state_weight = weight_hidden, weight_output in
  let bias_hidden = ref @@ Vector.zero n in
  let bias_output = ref @@ Vector.zero 1 in
  let bias = bias_hidden, bias_output in
  let elman = RNN_unit.elman_unit ~activations ~weight ~state_weight ~bias in
  let initial_state = ref @@ Vector.zero n in
  let initial_output = ref @@ Vector.zero 1 in
  let net = rnn elman initial_state initial_output in *)
  (* Jordan *)
  let activations = `Tanh,`Tanh,`None in
  let input_weight = ref @@ Vector.Linear.uniform 1 n in
  let output_weight = ref @@ Vector.Linear.uniform 1 n in
  let state_state = ref @@ Vector.Linear.uniform n n in
  let state_hidden = ref @@ Vector.Linear.uniform n n in
  let hidden_out = ref @@ Vector.Linear.uniform n 1 in
  let state_weight = state_state, state_hidden, hidden_out in
  let bias_state = ref @@ Vector.zero n in
  let bias_hidden = ref @@ Vector.zero n in
  let bias_output = ref @@ Vector.zero 1 in
  let bias = bias_state, bias_hidden, bias_output in
  let jordan = RNN_unit.jordan_unit ~activations ~input_weight ~output_weight ~state_weight ~bias in
  let initial_state = ref @@ Vector.zero n in
  let initial_output = ref @@ Vector.zero 1 in
  let net = rnn jordan initial_state initial_output in
  for _ = 0 to 1000 do
    input 
    |> net
    (* |> List.map (Backpropagatable.observe (fun x -> Printf.printf "value is %f\n%!" x.(0))) *)
    (* |> Backpropagatable.List.lift_error `Euclidean expected
    |> Backpropagatable.List.update error *)
    |> Backpropagatable.List.error `MSE expected
    |> Backpropagatable.descent 0.005
    |> Backpropagatable.run
  done;

  let k = 10 in
  let new_input = List.init k (fun _ -> Backpropagatable.cst [|1.0|]) in
  let new_expected = List.init k (fun n -> (1.0 /. (float (k-n))**2.)) in
  let pred = new_input |> net |> List.map (fun x -> (Backpropagatable.eval x).(0)) in
  List.iter2 (fun x -> Printf.printf "Value: %f, expected: %f\n" x) pred new_expected
