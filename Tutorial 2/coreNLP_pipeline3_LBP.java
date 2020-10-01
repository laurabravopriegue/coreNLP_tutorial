package edu.stanford.nlp.examples;

import java.io.File;
import java.util.Scanner;
import java.io.IOException;
import java.io.PrintWriter;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import java.util.*;
import edu.stanford.nlp.sentiment.*;
import edu.stanford.nlp.io.*;
import java.io.FileNotFoundException;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import org.ejml.simple.SimpleMatrix;

public class PipelineExample {

  public static void main(String[] args) throws IOException {

    // set up pipeline properties
    Properties props = new Properties();
    // set the list of annotators to run
    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,depparse,parse,sentiment");
    // build pipeline
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    //write output document
    System.out.println("Writing output document...");
    File file = new File("coreNLP_output.txt");
    //create the file if it doesn't exist
    if(!file.exists()){
      file.createNewFile();}

      PrintWriter out = new PrintWriter(file);

      //print column names on the output document
      out.println("par_id;sent_id;words;lemmas;posTags;nerTags;depParse;sentiment;veryneg;neg;neu;pos;verypos");

      //tries to open the input text documet
      try {
        System.out.println("Reading input text...");
        File myObj = new File("coreNLPinput_2.txt");
        Scanner myReader = new Scanner(myObj);

        //create sentence and paragraph ids
        int par_id = 1;
        int sent_id = 1;

        while (myReader.hasNextLine()) {
          String text = myReader.nextLine();
          System.out.println(text);

          // create a document object
          CoreDocument document = pipeline.processToCoreDocument(text);
          //annotate document
          pipeline.annotate(document);


          //get list of sentences
          System.out.println("Getting list of sentences in paragraph " + par_id);
          List <CoreSentence> sentences = document.sentences();

          //iterate through sentences
          System.out.println("Iterating through sentences...");


          for (CoreSentence sentence : sentences) {

            System.out.println("Processing sentence " + sent_id + " from paragraph " + par_id);
            System.out.println(sentence);

            //get list of tokens
            List<CoreLabel> tokens = sentence.tokens();

            //create list and fill it with the single token words
            List<String> words= new ArrayList();
            for (CoreLabel tok : tokens) {
              words.add(tok.word());}

              // get list of lemmas
              List<String> lemmas = sentence.lemmas();

              // get list of posTags
              List<String> posTags = sentence.posTags();

              // get list of ner tags
              List<String> nerTags = sentence.nerTags();

              // get dependency parsing graph
              SemanticGraph dependencyParse = sentence.dependencyParse();
              //turn it to compact string
              String depParse = dependencyParse.toCompactString();

              Tree tree = sentence.sentimentTree();
              //get overall score
              int sentimentScore =  RNNCoreAnnotations.getPredictedClass(tree);


              SimpleMatrix simpleMatrix = RNNCoreAnnotations.getPredictions(tree);

              //Gets prob for each sentiment using the elements of the sentimet matrix
                     float veryneg = (float)Math.round((simpleMatrix.get(0)*100d));
                     float neg = (float)Math.round((simpleMatrix.get(1)*100d));
                     float neutral = (float)Math.round((simpleMatrix.get(2)*100d));
                     float pos = (float)Math.round((simpleMatrix.get(3)*100d));
                     float verypos = (float)Math.round((simpleMatrix.get(4)*100d));


              System.out.println("Final score " + sentimentScore );
              System.out.println("Very neg prob " + veryneg + ", neg prob " + veryneg +  ", neutral prob " + neutral + ", pos prob " + pos  + ", very pos prob " + verypos);


              //printing data to the output document
              out.println(par_id + ";" + sent_id + ";" + words + ";" + lemmas + ";" + posTags + ";" + nerTags + ";" + depParse + ";" + sentimentScore + ";" + veryneg + ";" + neg + ";" + neutral + ";" + pos + ";" + verypos);

              sent_id += 1;

            }

            par_id += 1;

          }

          myReader.close();

        }

        //deal with error if input text is not found
        catch (FileNotFoundException e) {
          System.out.println("An error occurred.");
          e.printStackTrace();
        }

        out.close();
      }



    }
