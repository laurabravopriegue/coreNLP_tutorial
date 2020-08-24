package edu.stanford.nlp.examples;

import java.io.IOException;
import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ie.util.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.*;
import edu.stanford.nlp.trees.*;
import java.util.*;

public class PipelineExample {

    public static String text = "Marie was born in Paris.";

    public static void main(String[] args) throws IOException {
        // set up pipeline properties
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,depparse");
        // build pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        // create a document object
        CoreDocument document = pipeline.processToCoreDocument(text);
        pipeline.annotate(document);


        // get sentences of the document
        List <CoreSentence> sentences = document.sentences();
        System.out.println("Sentences of the document");
        System.out.println(sentences);
        System.out.println();

        //we iterate through the items of the list sentences

        for (CoreSentence sentence : sentences) {

          System.out.println("Sentence: " + sentence);
          System.out.println();

          //get tokens of the sentence
          List<CoreLabel> tokens = sentence.tokens();

          System.out.println("Tokens of the sentence:");
          for (CoreLabel tok : tokens) {
            //print the word
            System.out.println("Token: " + tok.word());
            //print the lemma
            System.out.println("Lemma: " + tok.lemma());
          }

          System.out.println();
          // list of the POS tags
          List<String> posTags = sentence.posTags();
          System.out.println("POS tags of the sentence:");
          System.out.println(posTags);
          System.out.println();

          // list of the ner tags
          List<String> nerTags = sentence.nerTags();
          System.out.println("POS tags of the sentence:");
          System.out.println(nerTags);
          System.out.println();

          // dependency parse for the second sentence
          SemanticGraph dependencyParse = sentence.dependencyParse();
          System.out.println("Dependency parsing of the sentence:");
          System.out.println("Root of the sentence: " + dependencyParse.getFirstRoot().word());
          System.out.println("List of dependencies: " + dependencyParse.toCompactString());
          System.out.println();

          }

    }

}
