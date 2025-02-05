(* Learning sums with RNN *)

let () = 
  let dataset_size = 5000 in
  let sequence_length = 2 in
  let max_num = 100 in
  Random.self_init ();
  let x, _ = Data.generate_string_data dataset_size sequence_length max_num in
  for i = 0 to 3 do
    Printf.printf "%d: %s|\n" i x.(i)
  done