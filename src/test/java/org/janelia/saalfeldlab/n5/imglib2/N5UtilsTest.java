/**
 * Copyright (c) 2017-2018, Stephan Saalfeld, Philipp Hanslovsky, Igor Pisarev
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
package org.janelia.saalfeldlab.n5.imglib2;

import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiFunction;

import net.imglib2.cache.img.CachedCellImg;
import net.imglib2.img.basictypeaccess.LongAccess;
import net.imglib2.img.basictypeaccess.ShortAccess;
import net.imglib2.img.basictypeaccess.volatiles.VolatileAccess;
import net.imglib2.type.operators.ValueEquals;
import net.imglib2.util.Util;
import org.hamcrest.CoreMatchers;
import org.hamcrest.MatcherAssert;
import org.janelia.saalfeldlab.n5.GzipCompression;
import org.janelia.saalfeldlab.n5.N5FSWriter;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;
import org.janelia.saalfeldlab.n5.RawCompression;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.util.Pair;
import net.imglib2.view.Views;

public class N5UtilsTest {

	static private String testDirPath = System.getProperty("user.home") + "/tmp/n5-imglib2-test";

	static private String datasetName = "/test/group/dataset";

	static private long[] dimensions = new long[]{11, 22, 33};

	static private int[] blockSize = new int[]{5, 7, 9};

	static short[] data;

	static private N5Writer n5;

	private static final int MAX_NUM_CACHE_ENTRIES = 10;

	private static final String EMPTY_DATASET = "/test/group/empty-dataset";

	private static final int EMPTY_BLOCK_VALUE = 123;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {

		final File testDir = new File(testDirPath);
		testDir.mkdirs();
		if (!(testDir.exists() && testDir.isDirectory()))
			throw new IOException("Could not create test directory for HDF5Utils test.");

		n5 = new N5FSWriter(testDirPath);

		final Random rnd = new Random();

		data = new short[(int)(dimensions[0] * dimensions[1] * dimensions[2])];
		for (int i = 0; i < data.length; ++i)
			data[i] = (short)rnd.nextInt();

		n5.createDataset(EMPTY_DATASET, dimensions, blockSize, N5Utils.dataType(new UnsignedShortType()), new GzipCompression());
	}

	@AfterClass
	public static void rampDownAfterClass() throws Exception {

		n5.remove("");
	}

	@Before
	public void setUp() throws Exception {}

	@After
	public void tearDown() throws Exception {}

	@Test
	public void testSaveAndOpen() throws InterruptedException, ExecutionException {

		final ArrayImg<UnsignedShortType, ?> img = ArrayImgs.unsignedShorts(data, dimensions);
		try {
			N5Utils.save(img, n5, datasetName, blockSize, new RawCompression());
			RandomAccessibleInterval<UnsignedShortType> loaded = N5Utils.open(n5, datasetName);
			for (final Pair<UnsignedShortType, UnsignedShortType> pair : Views
					.flatIterable(Views.interval(Views.pair(img, loaded), img)))
				Assert.assertEquals(pair.getA().get(), pair.getB().get());

			final ExecutorService exec = Executors.newFixedThreadPool(4);
			N5Utils.save(img, n5, datasetName, blockSize, new RawCompression(), exec);
			loaded = N5Utils.open(n5, datasetName);
			for (final Pair<UnsignedShortType, UnsignedShortType> pair : Views
					.flatIterable(Views.interval(Views.pair(img, loaded), img)))
				Assert.assertEquals(pair.getA().get(), pair.getB().get());
			exec.shutdown();

		} catch (final IOException e) {
			fail("Failed by I/O exception.");
			e.printStackTrace();
		}
	}

	@Test
	public void testOpenWithBoundedSoftRefCache() throws IOException {

		// existing dataset
		{
			final ArrayImg<UnsignedShortType, ?> img = ArrayImgs.unsignedShorts(data, dimensions);
			N5Utils.save(img, n5, datasetName, blockSize, new RawCompression());
			RandomAccessibleInterval<UnsignedShortType> loaded =
					N5Utils.openWithBoundedSoftRefCache(n5, datasetName, MAX_NUM_CACHE_ENTRIES);
			for (final Pair<UnsignedShortType, UnsignedShortType> pair : Views
					.flatIterable(Views.interval(Views.pair(img, loaded), img)))
				Assert.assertEquals(pair.getA().get(), pair.getB().get());
			MatcherAssert.assertThat(((CachedCellImg<UnsignedShortType, ?>)loaded).getAccessType(), CoreMatchers.instanceOf(ShortAccess.class));
		}

		// empty dataset with default value
		{
			RandomAccessibleInterval<UnsignedShortType> loaded =
					N5Utils.openWithBoundedSoftRefCache(n5, EMPTY_DATASET, MAX_NUM_CACHE_ENTRIES, new UnsignedShortType(EMPTY_BLOCK_VALUE));
			Views.iterable(loaded).forEach(val -> Assert.assertEquals(EMPTY_BLOCK_VALUE, val.get()));
			MatcherAssert.assertThat(((CachedCellImg<UnsignedShortType, ?>)loaded).getAccessType(), CoreMatchers.instanceOf(ShortAccess.class));
		}
	}

	@Test
	public void testVolatileOpenWithBoundedSoftRefCache() throws IOException {

		// existing dataset
		{
			final ArrayImg<UnsignedShortType, ?> img = ArrayImgs.unsignedShorts(data, dimensions);
			N5Utils.save(img, n5, datasetName, blockSize, new RawCompression());
			RandomAccessibleInterval<UnsignedShortType> loaded =
					N5Utils.openVolatileWithBoundedSoftRefCache(n5, datasetName, MAX_NUM_CACHE_ENTRIES);
			for (final Pair<UnsignedShortType, UnsignedShortType> pair : Views
					.flatIterable(Views.interval(Views.pair(img, loaded), img)))
				Assert.assertEquals(pair.getA().get(), pair.getB().get());
			Assert.assertEquals(UnsignedShortType.class, Util.getTypeFromInterval(loaded).getClass());
			MatcherAssert.assertThat(((CachedCellImg<UnsignedShortType, ?>)loaded).getAccessType(), CoreMatchers.instanceOf(VolatileAccess.class));
			MatcherAssert.assertThat(((CachedCellImg<UnsignedShortType, ?>)loaded).getAccessType(), CoreMatchers.instanceOf(ShortAccess.class));
		}

		// empty dataset with default value
		{
			RandomAccessibleInterval<UnsignedShortType> loaded =
					N5Utils.openVolatileWithBoundedSoftRefCache(n5, EMPTY_DATASET, MAX_NUM_CACHE_ENTRIES, new UnsignedShortType(EMPTY_BLOCK_VALUE));
			Views.iterable(loaded).forEach(val -> Assert.assertEquals(EMPTY_BLOCK_VALUE, val.get()));
			Assert.assertEquals(UnsignedShortType.class, Util.getTypeFromInterval(loaded).getClass());
			MatcherAssert.assertThat(((CachedCellImg<UnsignedShortType, ?>)loaded).getAccessType(), CoreMatchers.instanceOf(VolatileAccess.class));
			MatcherAssert.assertThat(((CachedCellImg<UnsignedShortType, ?>)loaded).getAccessType(), CoreMatchers.instanceOf(ShortAccess.class));
		}
	}
}
