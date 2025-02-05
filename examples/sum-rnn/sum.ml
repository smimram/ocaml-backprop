(* Learning sums with RNN *)
open Net
open Backprop.Net

let () = 
  let dataset_size = 5000 in
  let sequence_length = 2 in
  let max_num = 100 in
  Random.self_init ();
  let data = Data.generate_data dataset_size sequence_length max_num in
  for i = 0 to 1 do
    Array.iteri 
      (fun n x -> 
        Printf.printf "%d: [" n;
        Array.iter (fun x -> Printf.printf "%f " x) x;
        Printf.printf "]\n"
      )
      @@ fst data.(i);
    Array.iteri 
      (fun n x -> 
        Printf.printf "%d: [" n;
        Array.iter (fun x -> Printf.printf "%f " x) x;
        Printf.printf "]\n"
      )
      @@ snd data.(i)
  done;
  (* let epochs = 200 in *)
  let batch_size = 128 in
  let hidden_size = 128 in
  let learning_rate = 0.001 in
  let state1 = state hidden_size in
  let state2 = state hidden_size in
  let net = net hidden_size Data.vocabulary_size 
    (Array.length @@ snd data.(0)) ~state1 ~state2 in
  let batch = 
    Array.init batch_size 
      (fun _ ->
        let index = Random.int dataset_size in
        data.(index)) in
  let inputs = Array.map 
    (fun (x,_) -> 
      Array.map cst x
      ) batch in
  let expected = Array.map snd batch in
  inputs
  |> Array.map net
  |> Array.map2 
    (fun y x ->
      Array.map2 Vector.crossentropy y x
      )
    expected
  |> Array.map mux 
  |> Array.map Vector.sum 
  |> mux
  |> Vector.sum 
  |> descent learning_rate 
  |> run