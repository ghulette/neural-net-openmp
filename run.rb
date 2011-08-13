#!/usr/bin/env ruby

TRIALS = 1..30
THREADS = 0..1

UNIQ_STR = sprintf("%010d", rand * 100000000)
OUT_FILE = "out-#{UNIQ_STR}.txt" 
puts "Using file #{OUT_FILE}"


def parse_result(file)
  training = nil
  testing = nil
  File.open(file).each do |line| 
    if md = /Training: (\d+)/.match(line)
      training = md[1]
    end
    if md = /Testing: (\d+)/.match(line)
      testing = md[1]
    end
  end
  {:train => training, :test => testing}
end

results = Array.new
TRIALS.each do |trial|
  puts "Trial #{trial}/#{TRIALS.end}"
  trial_results = Hash.new
  THREADS.each do |n|
    puts "Running NN with #{n} threads"
    if(n == 0)
      `./nns > #{OUT_FILE}`
    else
      ENV['OMP_NUM_THREADS'] = n.to_s
      `./nnm > #{OUT_FILE}`
    end
    trial_results[n] = parse_result(OUT_FILE)
  end
  results[trial] = trial_results
end

File.delete(OUT_FILE)

puts
puts "Training Results"
puts "Trials,#{THREADS.map{|t|"#{t}"}.join(',')}"
TRIALS.each do |trial|
  set = Array.new
  THREADS.each do |nt|
    set << results[trial][nt][:train]
  end
  puts "#{trial},#{set.join(",")}"
end

puts
puts "Testing Results"
puts "Trials,#{THREADS.map{|t|"#{t}"}.join(',')}"
TRIALS.each do |trial|
  set = Array.new
  THREADS.each do |nt|
    set << results[trial][nt][:test]
  end
  puts "#{trial},#{set.join(",")}"
end

