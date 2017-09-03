/**
 * This file Copyright (c) 2011-2017 Magnolia International
 * Ltd.  (http://www.magnolia-cms.com). All rights reserved.
 *
 *
 * This file is dual-licensed under both the Magnolia
 * Network Agreement and the GNU General Public License.
 * You may elect to use one or the other of these licenses.
 *
 * This file is distributed in the hope that it will be
 * useful, but AS-IS and WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE, TITLE, or NONINFRINGEMENT.
 * Redistribution, except as permitted by whichever of the GPL
 * or MNA you select, is prohibited.
 *
 * 1. For the GPL license (GPL), you can redistribute and/or
 * modify this file under the terms of the GNU General
 * Public License, Version 3, as published by the Free Software
 * Foundation.  You should have received a copy of the GNU
 * General Public License, Version 3 along with this program;
 * if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * 2. For the Magnolia Network Agreement (MNA), this file
 * and the accompanying materials are made available under the
 * terms of the MNA which accompanies this distribution, and
 * is available at http://www.magnolia-cms.com/mna.html
 *
 * Any modifications to this file must keep this entire header
 * intact.
 *
 */
package info.magnolia.image.tagging;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import com.google.common.collect.Lists;

/**
 * Helper class with a static method that returns the label description.
 *
 * @author susaneraly
 */
public class ImageNetLabels {

    private final static String jsonUrl = "http://blob.deeplearning4j.org/utils/imagenet_class_index.json";
    private static ArrayList<String> predictionLabels = null;

    public ImageNetLabels() {
        this.predictionLabels = getLabels();
    }

    private static ArrayList<String> getLabels() {
        if (predictionLabels == null) {
            HashMap<String, ArrayList<String>> jsonMap;
            try {
                jsonMap = new ObjectMapper().readValue(new URL(jsonUrl), HashMap.class);
                predictionLabels = new ArrayList<>(jsonMap.size());
                for (int i = 0; i < jsonMap.size(); i++) {
                    predictionLabels.add(jsonMap.get(String.valueOf(i)).get(1));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return predictionLabels;
    }

    /**
     * Returns the description of tne nth class in the 1000 classes of ImageNet.
     * @param n
     * @return
     */
    public String getLabel(int n) {
        return predictionLabels.get(n);
    }

    /**
     * Given predictions from the trained model this method will return a string
     * listing the top five matches and the respective probabilities.
     *
     * @param predictions
     * @return
     */
    public List<Result> decodePredictions(INDArray predictions) {
        List<Result> predictionDescription = Lists.newArrayList();
        int[] top5 = new int[5];
        float[] top5Prob = new float[5];

        //brute force collect top 5
        int i = 0;
        for (int batch = 0; batch < predictions.size(0); batch++) {
            INDArray currentBatch = predictions.getRow(batch).dup();
            while (i < 5) {
                top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
                currentBatch.putScalar(0, top5[i], 0);
                predictionDescription.add(new Result(top5Prob[i] * 100, predictionLabels.get(top5[i])));
                i++;
            }
        }
        return predictionDescription;
    }

    /**
     * TODO: JavaDoc.
     */
    public class Result {
        private final float probability;
        private final String prediction;

        public Result(float probability, String prediction) {
            this.prediction = prediction;
            this.probability = probability;
        }

        public String getPrediction() {
            return prediction;
        }

        public float getProbability() {
            return probability;
        }
    }

}

