open Backprop
open Backpropagatable.Vector

let () = Random.self_init ()

let () =
  let m = 10 in
  let expected = List.init m (fun n -> Vector.scalar (sin (float (m-n-1)))) in
  (* let expected = List.init m (fun _ -> Vector.scalar 1.0) in *)
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
  (* let activations = `Relu,`Relu,`None in
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
  let net = rnn jordan initial_state initial_output in *)
  (* LSTM *)
  let wf = ref @@ Vector.Linear.uniform 1 n in
  let wi = ref @@ Vector.Linear.uniform 1 n in
  let wo = ref @@ Vector.Linear.uniform 1 n in
  let wc = ref @@ Vector.Linear.uniform 1 n in
  let weight = wf, wi, wo, wc in
  let uf = ref @@ Vector.Linear.uniform n n in
  let ui = ref @@ Vector.Linear.uniform n n in
  let uo = ref @@ Vector.Linear.uniform n n in
  let uc = ref @@ Vector.Linear.uniform n n in
  let output_weight = uf, ui, uo, uc in
  let biasf = ref @@ Vector.zero n in
  let biasi = ref @@ Vector.zero n in
  let biaso = ref @@ Vector.zero n in
  let biasc = ref @@ Vector.zero n in
  let bias = biasf, biasi, biaso, biasc in
  let lstm = RNN_unit.lstm_unit ~weight ~output_weight ~bias in
  let initial_state = ref @@ Vector.zero n in
  let initial_output = ref @@ Vector.zero n in
  let net = rnn lstm initial_state initial_output in
  let output_layer = 
    let weights = ref @@ Vector.Linear.uniform n 1 in
    let bias = ref @@ Vector.zero 1 in
    Backpropagatable.Vector.neural_network ~activation:`Sigmoid ~weights ~bias
  in
  for _ = 0 to 1000 do
    input 
    |> net
    |> List.map output_layer
    (* |> List.map (Backpropagatable.observe (fun x -> Printf.printf "value is %f\n%!" x.(0))) *)
    (* |> Backpropagatable.List.lift_error `Euclidean expected
    |> Backpropagatable.List.update error *)
    |> Backpropagatable.List.error `MSE expected
    |> Backpropagatable.descent 0.0001
    |> Backpropagatable.run
  done;

  let k = 35 in
  let new_input = List.init k (fun _ -> Backpropagatable.cst [|1.0|]) in
  let new_expected = List.init k (fun n -> sin (float (k-n-1))) in
  (* let new_expected = List.init k (fun _ -> 1.) in *)
  let pred = new_input |> net |> List.map (fun x -> (Backpropagatable.eval x).(0)) in
  List.iter2 (fun x -> Printf.printf "Value: %f, expected: %f\n" x) pred new_expected
