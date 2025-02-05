(* Learning sums with RNN *)

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
  done