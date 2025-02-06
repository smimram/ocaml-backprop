(* Learning sums with RNN *)
open Net
open Backprop.Net

let () = 
  let dataset_size = 2000 in
  let sequence_length = 2 in
  let max_num = 100 in
  Random.self_init ();
  let data = Data.generate_data dataset_size sequence_length max_num in
  let batch_size = 32 in
  let hidden_size = 32 in
  let learning_rate = 0.01 in
  let state1 = state hidden_size in
  let state2 = state hidden_size in
  let net = net hidden_size Data.vocabulary_size 
    (Array.length @@ snd data.(0)) ~state1 ~state2 in
  let epochs = 200 in
  for nb_epoch = 1 to epochs do
    Data.shuffle data;
    let start = ref 0 in
    let loss_av = ref 0. in
    let nb_batch = ref 0 in
    let total_samps = ref 0 in
    while !start < dataset_size do
      let len = min batch_size (dataset_size - !start) in
      let batch = Array.sub data !start len in
      start := !start + batch_size;
      let inputs = Array.map 
        (fun (x,_) -> 
          Array.map cst x
          ) batch in
      let expected = Array.map snd batch in
      let res = 
        inputs
        |> Array.map net
      in
      let loss =
        res
        |> Array.map2 
          (fun y x ->
            Array.map2 Vector.crossentropy y x
            )
          expected
        |> Array.map mux 
        |> Array.map Vector.sum 
        |> mux
        |> Vector.sum in
      loss_av := !loss_av +. (eval loss);
      loss
      |> descent learning_rate 
      |> run;
      incr nb_batch;
      total_samps := !total_samps+len;
      Printf.printf "\rEpoch %d/%d: Batch %d done, Loss: %f%!" 
        nb_epoch epochs
        !nb_batch (!loss_av/.(float !total_samps));
    done;
    Printf.printf "\n";
  done;

  let test = Data.generate_data 1 sequence_length max_num in
  let input, expected = test.(0) in
  let res = 
    input
    |> Array.map cst
    |> net in
  let pred = res |> Array.map eval in
  Printf.printf "Input:\n";
  Array.iter 
    (fun x -> 
      Printf.printf "%s\n" @@ Backprop.Vector.to_string x) 
    input;
  Printf.printf "Expected:\n";
  Array.iter 
    (fun x -> 
      Printf.printf "%s\n" @@ Backprop.Vector.to_string x) 
    expected;
  Printf.printf "Probability:\n";
  Array.iter 
    (fun x -> 
      Printf.printf "%s\n" @@ Backprop.Vector.to_string x) 
    pred;
  let loss =
    res
    |> Array.map2 Vector.crossentropy expected
    |> mux 
    |> Vector.sum 
    |> eval in
  Printf.printf "Loss: %f" loss